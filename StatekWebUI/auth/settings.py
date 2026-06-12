from . import OIDCSettingsBase


class StatekUISettings(OIDCSettingsBase):
    """OIDC settings for the Statek Dashboard, loaded from environment / .env_statek.

    Default provider: AWS Cognito.
    Set oidc_well_known_url to:
        https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/openid-configuration

    Access is restricted to users in the ``super-admin`` Cognito group.

    Cognito super-admin group setup
    --------------------------------

    1. Open the AWS Cognito Console and navigate to your User Pool.

    2. Create the ``super-admin`` group:
       - Go to **Groups** tab → **Create group**.
       - Group name: ``super-admin``
       - (IAM role and precedence can be left empty.)
       - Click **Create group**.

    3. Add users to the group:
       - Go to **Users** tab → select the user → **Group memberships** → **Add to group**.
       - Select ``super-admin`` → **Add**.

    4. Ensure the group claim is in the ID token (enabled by default):
       - Go to **App integration** → **App client settings** for the statek-UI client.
       - Under **Hosted UI** → **OpenID Connect scopes**, ensure ``openid`` is selected.
       - Cognito automatically includes ``cognito:groups`` in the ID token when the
         user belongs to at least one group.  No custom token configuration is needed.

    5. Set the following environment variables (in ``.env_statek`` or ``.env_local_statek``):

       .. code-block:: bash

           OIDC_WELL_KNOWN_URL=https://cognito-idp.<region>.amazonaws.com/<pool-id>/.well-known/openid-configuration
           OIDC_CLIENT_ID=<app-client-id>
           OIDC_CLIENT_SECRET=<app-client-secret>
           OIDC_REDIRECT_URI=https://<statek-ui-domain>/auth/callback
           OIDC_POST_LOGOUT_URI=/
           OIDC_LOGOUT_URL=https://<cognito-domain>.auth.<region>.amazoncognito.com/logout
           COOKIE_SECRET_KEY=<random-hex-string>

    To change the required group name, set ``REQUIRED_COGNITO_GROUP`` in the env file
    (default: ``super-admin``).
    """

    model_config = {"env_file": [".env_statek", ".env_local_statek"], "extra": "ignore"}
