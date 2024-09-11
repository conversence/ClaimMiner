from . import spa_router, Request, get_base_template_vars, templates

from . import auth_routes
from . import collection
from . import docs
from . import scatterplot
from . import claim_clusters
from . import search
from . import claim
from . import dashboard
from . import prompts
from . import analysis
from . import triggers
from . import task_template

from ..auth import active_user_c_dep


@spa_router.get("/{path:path}")
async def spa(request: Request, current_user: active_user_c_dep, path: str):
    base_vars = await get_base_template_vars(request, current_user)
    return templates.TemplateResponse(request, "spa.html", base_vars | dict(path=path))
