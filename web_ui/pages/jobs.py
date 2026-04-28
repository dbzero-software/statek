"""All Jobs list page for the Statek web UI."""

import dbzero as db0

from web_ui.nicegui_compat import ui
from web_ui.components.status_badge import create_status_badge
from web_ui.model_bindings import get_all_jobs
from web_ui.pages.job_detail import create_job_detail_dialog

PAGE_SIZE = 20


def _paginate(items: list, page: int, page_size: int) -> list:
    """Return the slice of *items* for the given 1-indexed *page*."""
    start = (page - 1) * page_size
    return items[start:start + page_size]


def _job_agent_role(job) -> str:
    """Return the agent role label for a job, or '—' on any error."""
    try:
        if job.job_def is None:
            return '—'
        return job.job_def.agent.role if job.job_def.agent is not None else '—'
    except Exception:  # pylint: disable=broad-except
        return '—'


def _job_status_str(job) -> str:
    """Return the status string for a job, or '—' on any error."""
    try:
        return str(job.status) if job.status is not None else '—'
    except Exception:  # pylint: disable=broad-except
        return '—'


def _job_uuid(job) -> str:
    """Return the UUID string for a job, or '—' if not available."""
    try:
        return str(db0.uuid(job))
    except Exception:  # pylint: disable=broad-except
        return '—'


def _job_model(job) -> str:
    """Return the job definition model label for a job, or '—' on any error."""
    try:
        if job.job_def is None or job.job_def.model is None:
            return '—'
        return str(job.job_def.model)
    except Exception:  # pylint: disable=broad-except
        return '—'


def _job_tps_label(job) -> str:
    """Return formatted tokens-per-second label for a job, or '—' on any error."""
    try:
        tps = job.tokens_per_sec()
        return f'{tps:.0f} tok/s' if tps else '—'
    except Exception:  # pylint: disable=broad-except
        return '—'


def _render_job_row(job) -> None:
    """Render a single job as a table-style row."""
    uuid_str = _job_uuid(job)
    agent_role = _job_agent_role(job)
    status_str = _job_status_str(job)
    model = _job_model(job)
    num_turns = getattr(job, 'num_turns', 0)
    total_cost = job.usage.total_cost or 0.0
    cost_label = f'${total_cost:.4f}' if total_cost else '—'
    tps_label = _job_tps_label(job)

    with ui.row().classes(
        'w-full items-center gap-4 px-4 py-2 border-b border-gray-100 hover:bg-gray-50'
        ' cursor-pointer'
    ).on('click', lambda j=job: create_job_detail_dialog(j)):
        ui.label(uuid_str).classes('text-xs font-mono text-gray-400 w-72 shrink-0 truncate')
        ui.label(agent_role).classes('text-sm font-medium text-gray-800 flex-1 truncate')
        create_status_badge(status_str)
        ui.label(model).classes('text-xs text-gray-500 font-mono w-40 shrink-0 truncate')
        ui.label(str(num_turns)).classes('text-xs text-gray-600 w-12 text-right shrink-0')
        ui.label(cost_label).classes('text-xs text-gray-600 font-mono w-20 text-right shrink-0')
        ui.label(tps_label).classes('text-xs text-gray-600 font-mono w-24 text-right shrink-0')
        ui.icon('open_in_full').classes('text-xs text-gray-300 shrink-0')


def create_jobs_page() -> None:
    """Render the All Jobs paginated list page."""
    all_jobs = list(get_all_jobs())
    total = len(all_jobs)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    page_state = {'page': 1}

    with ui.row().classes('w-full items-center justify-between mb-4'):
        ui.label('All Jobs').classes('text-2xl font-bold text-gray-900')
        ui.badge(str(total), color='primary').classes('text-sm')

    if not all_jobs:
        with ui.card().classes('w-full'):
            ui.label('No jobs found in the database.').classes('text-gray-500 italic')
        return

    @ui.refreshable
    def jobs_list() -> None:
        page_jobs = _paginate(all_jobs, page_state['page'], PAGE_SIZE)

        with ui.card().classes('w-full shadow-sm p-0 overflow-hidden'):
            # Header row
            with ui.row().classes(
                'w-full items-center gap-4 px-4 py-2 bg-gray-50 border-b border-gray-200'
            ):
                ui.label('UUID').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide w-72 shrink-0'
                )
                ui.label('Agent').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide flex-1'
                )
                ui.label('Status').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide w-24 shrink-0'
                )
                ui.label('Model').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide w-40 shrink-0'
                )
                ui.label('Turns').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide w-12 text-right shrink-0'
                )
                ui.label('Cost').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide w-20 text-right shrink-0'
                )
                ui.label('Speed').classes(
                    'text-xs font-semibold text-gray-500 uppercase tracking-wide w-24 text-right shrink-0'
                )

            for job in page_jobs:
                _render_job_row(job)

        # Pagination controls
        if total_pages > 1:
            with ui.row().classes('w-full items-center justify-between mt-3'):
                def _prev(p=page_state):
                    if p['page'] > 1:
                        p['page'] -= 1
                        jobs_list.refresh()

                def _next(p=page_state):
                    if p['page'] < total_pages:
                        p['page'] += 1
                        jobs_list.refresh()

                ui.button('Previous', icon='chevron_left', on_click=_prev).props('flat dense')
                ui.label(f'Page {page_state["page"]} of {total_pages}').classes(
                    'text-sm text-gray-600'
                )
                ui.button('Next', icon='chevron_right', on_click=_next).props('flat dense')

    jobs_list()
