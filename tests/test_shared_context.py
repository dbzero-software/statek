"""Tests for shared context data structures."""

# pylint: disable=no-member,protected-access

from datetime import datetime, timedelta, timezone

import pytest
import dbzero as db0

from statek.executors.job import Job, JobStatus, parse_warmup_code
from statek.executors.utils import run_job_step
from statek.shared_context import (
    ContextCategory,
    ContextCategoryDict,
    ContextVar,
    SharedContext,
    ContextFallbackProxy,
    SharedContextProxy,
    init_shared_context,
    print_locals,
    shared_context_set_var,
)
from statek.utils import _statek_ctx_scope


pytestmark = pytest.mark.usefixtures("db0_fixture")


def test_context_category_dict_has_default_categories():
    """The shared category dictionary is initialized with canonical defaults."""
    categories = ContextCategoryDict()

    assert set(categories.categories) == {"PREFERENCE", "ENTITY", "VOCABULARY"}
    assert categories.get("PREFERENCE").name == "PREFERENCE"


def test_context_category_dict_is_singleton_and_lookup_is_case_insensitive():
    """All callers share one dictionary and legacy lowercase names resolve."""
    first = ContextCategoryDict()
    second = ContextCategoryDict()

    assert first is second
    assert first.get("preference") is first.get("PREFERENCE")
    assert first.get("unknown") is None


def test_context_category_accepts_prefix():
    """Context categories can be persisted in an explicitly selected prefix."""
    category = ContextCategory(name="CUSTOM", prefix="category-prefix")

    assert db0.get_prefix_of(category).name == "category-prefix"
    assert category.name == "CUSTOM"


def test_context_category_dict_accepts_prefix_as_scoped_singleton():
    """Each explicit prefix owns an independent category dictionary singleton."""
    categories = ContextCategoryDict(prefix="categories-prefix")

    assert db0.get_prefix_of(categories).name == "categories-prefix"
    assert ContextCategoryDict(prefix="categories-prefix") is categories
    assert db0.find_singleton(
        ContextCategoryDict, prefix="categories-prefix"
    ) is categories
    assert all(
        db0.get_prefix_of(category).name == "categories-prefix"
        for category in categories.categories.values()
    )

    positional = ContextCategoryDict("positional-prefix")
    assert db0.get_prefix_of(positional).name == "positional-prefix"


def test_context_category_dict_preserves_custom_categories_with_prefix():
    """A caller-provided mapping remains supported alongside prefix selection."""
    custom = ContextCategory(name="CUSTOM", prefix="custom-prefix")

    categories = ContextCategoryDict(
        categories={"CUSTOM": custom},
        prefix="custom-prefix",
    )

    assert categories.categories == {"CUSTOM": custom}
    assert categories.get("CUSTOM") is custom


def test_set_var_ignores_unknown_category():
    """Variables outside the shared category dictionary are not stored."""
    context = SharedContext()

    context.set_var("unknown", "key", "value", "description")

    assert "key" not in context
    assert context.get_var("key") is None


def test_shared_context_accepts_prefix():
    """Shared contexts can be persisted in an explicitly selected prefix."""
    context = SharedContext(prefix="context-prefix")

    assert db0.get_prefix_of(context).name == "context-prefix"

    positional = SharedContext("positional-context-prefix")
    assert db0.get_prefix_of(positional).name == "positional-context-prefix"


def test_shared_context_selects_variables_by_category():
    """Select returns only variables belonging to the requested category."""
    context = SharedContext()
    context.set_var("preference", "tone", "concise", "Preferred tone")
    context.set_var("entity", "client", "Acme", "Current client")
    category = ContextCategoryDict().get("PREFERENCE")

    selected = list(context.select(category))

    assert [var.value for var in selected] == ["concise"]
    assert selected[0].use_count == 1


def test_shared_context_select_applies_optional_filter():
    """Select applies a ContextVar predicate after category matching."""
    context = SharedContext()
    context.set_var("preference", "tone", "concise", "Preferred tone")
    context.set_var("preference", "language", "Polish", "Preferred language")
    category = ContextCategoryDict().get("PREFERENCE")

    selected = list(context.select(category, filter=lambda var: var.value == "Polish"))

    assert [var.value for var in selected] == ["Polish"]
    assert context._peek_var("tone").use_count == 0


def test_shared_context_proxy_select_resolves_latest_duplicate_key():
    """Proxy selection returns one newest resolvable variable per key."""
    older_time = datetime.now(timezone.utc)
    newer_time = older_time + timedelta(seconds=1)
    writable = SharedContext()
    readable = SharedContext()
    older = _store_var_at(writable, "tone", "concise", older_time)
    newer = _store_var_at(readable, "tone", "detailed", newer_time)
    writable.set_var("preference", "language", "Polish", "Preferred language")
    proxy = SharedContextProxy(writable=writable, readables=readable)
    category = ContextCategoryDict().get("PREFERENCE")

    selected = list(proxy.select(category))

    assert {var.value for var in selected} == {"detailed", "Polish"}
    assert newer.use_count == 1
    assert older.use_count == 0


def test_init_shared_context_is_hidden_system_tool():
    """Context initialization executes internally without LLM exposure."""
    assert init_shared_context.tool_system is True
    assert init_shared_context.tool_hidden is True


@pytest.mark.asyncio
async def test_hidden_shared_context_warmup_executes_without_history_exposure(
    job_def_factory,
):
    """Hidden LTM warmup initializes context and stays out of LLM-facing history."""
    parsed_warmup = parse_warmup_code(
        "#STATEK: hidden = True\n"
        "init_shared_context(user)\n"
        "# ----------\n"
        "visible_value = 1"
    )
    job_def = job_def_factory(warmup_code=parsed_warmup)
    job = Job(job_def=job_def, job_status=JobStatus.READY)
    job.py_env.local_state["user"] = "user-1"

    await run_job_step(job)

    history_text = "\n".join(str(item.content) for item in job.get_chat_history())
    assert job._get_shared_context() is not None
    assert "init_shared_context" not in history_text
    assert "user-1" not in history_text


def test_init_shared_context_requires_current_job():
    """Initialization cannot bind a context without an active job."""
    with pytest.raises(RuntimeError, match="current job"):
        init_shared_context("user")


def test_init_shared_context_reuses_exact_binding_across_jobs(job_factory):
    """Jobs initialized with the same specifiers share one persisted context."""
    first_job = job_factory()
    second_job = job_factory()

    with _statek_ctx_scope({"job": first_job}):
        init_shared_context("user")
    with _statek_ctx_scope({"job": second_job}):
        init_shared_context("user")

    assert first_job._get_shared_context() is second_job._get_shared_context()


def test_init_shared_context_merges_subset_contexts(job_factory):
    """A narrower binding inherits broader contexts for reads."""
    broad_job = job_factory()
    narrow_job = job_factory()
    with _statek_ctx_scope({"job": broad_job}):
        init_shared_context("user")
    broad_job._get_shared_context().set_var(
        "preference", "tone", "concise", "Preferred tone"
    )

    with _statek_ctx_scope({"job": narrow_job}):
        init_shared_context("department", "user")

    active = narrow_job._get_shared_context()
    assert isinstance(active, SharedContextProxy)
    assert active.get_var("tone").value == "concise"


def test_init_shared_context_binding_is_order_independent(job_factory):
    """Specifier order and duplicates do not alter exact binding identity."""
    first_job = job_factory()
    second_job = job_factory()
    with _statek_ctx_scope({"job": first_job}):
        init_shared_context("user", "department", "user")
    with _statek_ctx_scope({"job": second_job}):
        init_shared_context("department", "user")

    first_job._get_shared_context().set_var(
        "entity", "client", "Acme", "Current client"
    )
    assert second_job._get_shared_context().get_var("client").value == "Acme"


def test_init_shared_context_accepts_memo_specifiers(job_factory):
    """Persistent memo objects can identify a shared context."""
    job = job_factory()
    specifier = ContextCategoryDict().get("ENTITY")

    with _statek_ctx_scope({"job": job}):
        init_shared_context(specifier)

    assert isinstance(job._get_shared_context(), SharedContext)


def test_init_shared_context_rejects_invalid_specifiers(job_factory):
    """Only strings and dbzero memo objects can identify contexts."""
    job = job_factory()

    with _statek_ctx_scope({"job": job}):
        with pytest.raises(TypeError, match="specifier"):
            init_shared_context(123)


def test_init_shared_context_read_only_rejects_writes(job_factory):
    """Read-only initialization exposes context values but rejects mutation."""
    job = job_factory()
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user", read_only=True)

    active = job._get_shared_context()
    assert isinstance(active, SharedContextProxy)
    with pytest.raises(PermissionError, match="read-only"):
        active.set_var("preference", "tone", "concise", "Preferred tone")


def test_shared_context_set_var_is_visible_system_tool():
    """Agents can discover the shared-context write utility."""
    assert shared_context_set_var.tool_system is True
    assert shared_context_set_var.tool_hidden is False


def test_print_locals_is_visible_system_tool():
    """Agents can discover the shared-context inspection tool."""
    assert print_locals.tool_system is True
    assert print_locals.tool_hidden is False


def test_print_locals_prints_shared_context_variables(job_factory, capsys):
    """The tool prints matching variables from the job's shared context."""
    job = job_factory()
    preference = ContextCategoryDict().get("PREFERENCE")
    entity = ContextCategoryDict().get("ENTITY")
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user")
        shared_context_set_var(
            preference, "colors", "blue", "preferred calendar palette"
        )
        shared_context_set_var(entity, "client", "Acme", "current client")
        print_locals("PREFERENCE")

    assert capsys.readouterr().out == (
        "Locals from the category: PREFERENCE\n"
        "colors  # PREFERENCE: preferred calendar palette\n"
    )


def test_print_locals_filters_shared_values_by_any_type(job_factory, capsys):
    """Positional type filters use any-of matching on shared values."""
    job = job_factory()
    entity = ContextCategoryDict().get("ENTITY")
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user")
        shared_context_set_var(entity, "count", 42, "numeric entity")
        shared_context_set_var(entity, "client", "Acme", "named entity")
        print_locals("ENTITY", int)

    assert capsys.readouterr().out == (
        "Locals from the category: ENTITY, type: int\n"
        "count  # ENTITY: numeric entity\n"
    )


def test_shared_context_set_var_forwards_to_current_job_context(job_factory):
    """The utility stores a variable in the initialized active context."""
    job = job_factory()
    category = ContextCategoryDict().get("PREFERENCE")
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user")
        shared_context_set_var(category, "tone", "concise", "Preferred tone")

    var = job._get_shared_context().get_var("tone")
    assert var is not None
    assert var.value == "concise"


def test_shared_context_set_var_without_job_is_noop():
    """The utility does nothing when called outside a job."""
    category = ContextCategoryDict().get("PREFERENCE")

    assert shared_context_set_var(
        category, "tone", "concise", "Preferred tone"
    ) is None


def test_shared_context_set_var_without_initialized_context_is_noop(job_factory):
    """The utility does nothing until shared context has been initialized."""
    job = job_factory()
    category = ContextCategoryDict().get("PREFERENCE")

    with _statek_ctx_scope({"job": job}):
        assert shared_context_set_var(
            category, "tone", "concise", "Preferred tone"
        ) is None


def test_shared_context_local_context_resolves_missing_variable(job_factory):
    """Missing local names resolve lazily without becoming local entries."""
    job = job_factory()
    category = ContextCategoryDict().get("PREFERENCE")
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user")
        shared_context_set_var(category, "tone", "concise", "Preferred tone")

    local_context = ContextFallbackProxy({}, job._get_shared_context())
    assert local_context["tone"] == "concise"
    assert "tone" not in local_context


def test_shared_context_local_context_prefers_true_local_variable(job_factory):
    """An existing local shadows a shared variable and remains writable."""
    job = job_factory()
    category = ContextCategoryDict().get("PREFERENCE")
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user")
        shared_context_set_var(category, "tone", "concise", "Preferred tone")

    local_context = ContextFallbackProxy(
        {"tone": "local"}, job._get_shared_context()
    )
    local_context["tone"] = "changed"
    assert local_context["tone"] == "changed"


def test_shared_context_local_context_rejects_shared_variable_write(job_factory):
    """Writing a fallback-resolved shared variable raises PermissionError."""
    job = job_factory()
    category = ContextCategoryDict().get("PREFERENCE")
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user")
        shared_context_set_var(category, "tone", "concise", "Preferred tone")

    local_context = ContextFallbackProxy({}, job._get_shared_context())
    with pytest.raises(PermissionError, match="tone"):
        local_context["tone"] = "detailed"
    assert "tone" not in local_context


@pytest.mark.parametrize("operation", ["update", "setdefault", "ior"])
def test_shared_context_local_context_blocks_all_add_paths(job_factory, operation):
    """Dict mutation helpers cannot bypass shared-name collision protection."""
    job = job_factory()
    category = ContextCategoryDict().get("PREFERENCE")
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user")
        shared_context_set_var(category, "tone", "concise", "Preferred tone")

    local_context = ContextFallbackProxy({}, job._get_shared_context())
    with pytest.raises(PermissionError, match="tone"):
        if operation == "update":
            local_context.update({"safe": 1, "tone": "detailed"})
        elif operation == "setdefault":
            local_context.setdefault("tone", "detailed")
        else:
            local_context |= {"tone": "detailed"}

    assert "tone" not in local_context
    assert "safe" not in local_context


def test_shared_context_local_context_ignores_removed_category(job_factory):
    """Only variables whose category remains registered are recognized."""
    job = job_factory()
    categories = ContextCategoryDict()
    category = categories.get("PREFERENCE")
    with _statek_ctx_scope({"job": job}):
        init_shared_context("user")
        shared_context_set_var(category, "tone", "concise", "Preferred tone")
    del categories.categories["PREFERENCE"]

    local_context = ContextFallbackProxy({}, job._get_shared_context())
    with pytest.raises(KeyError, match="tone"):
        _ = local_context["tone"]


def test_context_var_defaults_created_at_and_use_count():
    """ContextVar records creation time and starts unused."""
    var = ContextVar(category="preference", value="concise", description="tone")

    assert var.category == "preference"
    assert var.value == "concise"
    assert var.description == "tone"
    assert isinstance(var.created_at, datetime)
    assert var.created_at.tzinfo is timezone.utc
    assert var.use_count == 0


def test_set_and_get_var_returns_context_var():
    """SharedContext stores named variables and returns ContextVar objects."""
    context = SharedContext()

    context.set_var("preference", "tone", "concise", "Preferred response style")
    var = context.get_var("tone")

    assert var is not None
    assert var.category == "preference"
    assert var.value == "concise"
    assert var.description == "Preferred response style"
    assert var.use_count == 1


def test_get_missing_var_returns_none():
    """Missing variables return None without creating a context entry."""
    context = SharedContext()

    assert context.get_var("missing") is None
    assert "missing" not in context


def test_contains_reports_existing_key_without_incrementing_use_count():
    """Membership checks do not count as variable use."""
    context = SharedContext()
    context.set_var("entity", "client", "Acme", "Current client")

    assert "client" in context
    var = context.get_var("client")

    assert var is not None
    assert var.use_count == 1


def test_get_var_increments_use_count_each_time():
    """Successful lookups increment use_count on the stored variable."""
    context = SharedContext()
    context.set_var("vocabulary", "crm", "customer relationship manager", "Term expansion")

    first = context.get_var("crm")
    second = context.get_var("crm")

    assert first is second
    assert second is not None
    assert second.use_count == 2


def test_set_var_overwrites_existing_key_with_new_context_var():
    """Setting an existing key replaces value, metadata, timestamp, and use count."""
    context = SharedContext()
    context.set_var("preference", "tone", "concise", "Initial style")
    first = context.get_var("tone")
    assert first is not None

    context.set_var("preference", "tone", "detailed", "Updated style")
    second = context.get_var("tone")

    assert second is not None
    assert second is not first
    assert second.value == "detailed"
    assert second.description == "Updated style"
    assert second.use_count == 1
    assert second.created_at >= first.created_at


def test_context_instances_do_not_share_variables():
    """Each SharedContext instance owns an independent variable dictionary."""
    first = SharedContext()
    second = SharedContext()

    first.set_var("entity", "client", "Acme", "Current client")

    assert "client" in first
    assert "client" not in second
    assert second.get_var("client") is None


def _store_var_at(context: SharedContext, key: str, value: str, created_at: datetime) -> ContextVar:
    """Store a context variable with a deterministic timestamp for merge tests."""
    context.set_var("preference", key, value, f"{key} description")
    var = context.get_var(key)
    assert var is not None
    var.created_at = created_at
    var.use_count = 0
    return var


def test_proxy_set_var_forwards_to_writable_context():
    """SharedContextProxy writes only to its writable context."""
    writable = SharedContext()
    readable = SharedContext()
    proxy = SharedContextProxy(writable=writable, readables=readable)

    proxy.set_var("preference", "tone", "concise", "Preferred response style")

    assert "tone" in writable
    assert "tone" not in readable
    var = proxy.get_var("tone")
    assert var is not None
    assert var.value == "concise"
    assert var.use_count == 1


def test_proxy_get_var_reads_from_writable_without_readables():
    """SharedContextProxy works with only a writable context."""
    writable = SharedContext()
    writable.set_var("preference", "tone", "concise", "Preferred response style")
    proxy = SharedContextProxy(writable=writable)

    var = proxy.get_var("tone")

    assert var is not None
    assert var.value == "concise"
    assert var.use_count == 1


def test_proxy_get_var_reads_from_single_readable_context():
    """SharedContextProxy reads variables from its readable context."""
    writable = SharedContext()
    readable = SharedContext()
    readable.set_var("entity", "client", "Acme", "Current client")
    proxy = SharedContextProxy(writable=writable, readables=readable)

    var = proxy.get_var("client")

    assert var is not None
    assert var.value == "Acme"
    assert var.use_count == 1


def test_proxy_get_var_reads_from_multiple_readable_contexts():
    """SharedContextProxy merges reads from multiple readable contexts."""
    writable = SharedContext()
    first_readable = SharedContext()
    second_readable = SharedContext()
    first_readable.set_var("entity", "client", "Acme", "Current client")
    second_readable.set_var("vocabulary", "crm", "customer relationship manager", "Term")
    proxy = SharedContextProxy(writable=writable, readables=[first_readable, second_readable])

    client = proxy.get_var("client")
    crm = proxy.get_var("crm")

    assert client is not None
    assert client.value == "Acme"
    assert crm is not None
    assert crm.value == "customer relationship manager"


def test_proxy_get_var_returns_latest_duplicate_key():
    """When duplicate keys exist, SharedContextProxy returns the latest created_at."""
    older_time = datetime.now(timezone.utc)
    newer_time = older_time + timedelta(seconds=1)
    writable = SharedContext()
    readable = SharedContext()
    older = _store_var_at(writable, "tone", "concise", older_time)
    newer = _store_var_at(readable, "tone", "detailed", newer_time)
    proxy = SharedContextProxy(writable=writable, readables=readable)

    var = proxy.get_var("tone")

    assert var is newer
    assert var.value == "detailed"
    assert newer.use_count == 1
    assert older.use_count == 0


def test_proxy_get_var_prefers_writable_when_created_at_ties():
    """SharedContextProxy returns writable variable when timestamps tie."""
    created_at = datetime.now(timezone.utc)
    writable = SharedContext()
    readable = SharedContext()
    writable_var = _store_var_at(writable, "tone", "concise", created_at)
    readable_var = _store_var_at(readable, "tone", "detailed", created_at)
    proxy = SharedContextProxy(writable=writable, readables=readable)

    var = proxy.get_var("tone")

    assert var is writable_var
    assert var.value == "concise"
    assert writable_var.use_count == 1
    assert readable_var.use_count == 0


def test_proxy_contains_checks_all_contexts_without_incrementing_use_count():
    """SharedContextProxy membership checks do not count as variable use."""
    writable = SharedContext()
    readable = SharedContext()
    readable.set_var("entity", "client", "Acme", "Current client")
    proxy = SharedContextProxy(writable=writable, readables=readable)

    assert "client" in proxy
    assert "missing" not in proxy
    var = proxy.get_var("client")
    assert var is not None
    assert var.use_count == 1


def test_proxy_contains_finds_key_in_writable_context():
    """SharedContextProxy membership checks include the writable context."""
    writable = SharedContext()
    readable = SharedContext()
    writable.set_var("preference", "tone", "concise", "Preferred response style")
    proxy = SharedContextProxy(writable=writable, readables=readable)

    assert "tone" in proxy
    var = proxy.get_var("tone")
    assert var is not None
    assert var.use_count == 1
