import inspect
from abc import ABC
from typing import Any, Callable, List, Optional, Union, get_origin, get_args

from fastapi import APIRouter, FastAPI
from fastapi.openapi.utils import get_openapi

# Import authorization-related modules has been moved inside methods for on-demand import


def _create_route_decorator(http_method: str) -> Callable:
    """
    Internal helper function to create FastAPI route decorators (get, post, put, delete, etc.).

    Args:
        http_method (str): HTTP method name (e.g., "GET", "POST").

    Returns:
        Callable: A decorator that accepts path and other parameters for APIRouter.add_api_route.
    """

    def decorator(
        path: str, extra_models: Optional[List[Any]] = None, **kwargs: Any
    ) -> Callable:
        """
        response_class: Determines the **"transport method" and "underlying type"** of the response.
        It controls which class FastAPI uses to package and send the HTTP response.
        response_model: Determines the **"data structure" and "validation rules"** of the response body.
        It is used for data filtering, format conversion, and automatically generating schemas in API documentation.
        summary and responses: Used entirely for **"API documentation (OpenAPI)"**.
        They do not affect any runtime behavior but enrich and refine the generated documentation (e.g., Swagger UI or ReDoc).
        """

        def wrapper(func: Callable) -> Callable:
            # Use a special attribute to mark the function and store routing information
            # This avoids a global registry, making each controller self-contained
            setattr(func, "__route_info__", (path, [http_method], kwargs))
            # Store extra_models for later OpenAPI generation
            setattr(func, "__extra_models__", extra_models or [])
            return func

        return wrapper

    return decorator


get = _create_route_decorator("GET")
post = _create_route_decorator("POST")
put = _create_route_decorator("PUT")
delete = _create_route_decorator("DELETE")
patch = _create_route_decorator("PATCH")
head = _create_route_decorator("HEAD")
options = _create_route_decorator("OPTIONS")


class BaseController(ABC):
    """
    Base controller class that supports automatic route registration via decorators.

    Inherit from this class and use @get, @post, etc., decorators to define your API endpoints.
    During initialization, the controller automatically collects all decorated routes.

    Usage example:
    ```python
    # a_controller.py
    from .base_controller import BaseController, get

    class UserController(BaseController):
        def __init__(self):
            super().__init__(prefix="/users", tags=["Users"])

        @get("/")
        def list_users(self):
            return [{"id": 1, "name": "user1"}]

    # app.py
    from fastapi import FastAPI
    from .a_controller import UserController

    app = FastAPI()
    controllers = [UserController()] # "Scanned" list of controllers

    for controller in controllers:
        controller.register_to_app(app)
    ```
    """

    # Class-level security configuration provider; subclasses can override this attribute
    _security_config_provider: Optional[Callable[[], List[dict]]] = None

    def __init__(
        self,
        prefix: str = "",
        tags: Optional[List[str]] = None,
        default_auth: str = "require_user",
        **kwargs: Any,
    ):
        """
        Initialize the controller.

        Args:
            prefix (str, optional): Common prefix for all routes under this controller.
            tags (Optional[List[str]], optional): Tags used for grouping in OpenAPI documentation.
            default_auth (str, optional): Default authorization strategy. Possible values:
                - "require_user": Requires user authentication by default (default)
                - "require_anonymous": Allows anonymous access by default
                - "require_admin": Requires admin privileges by default
                - "require_signature": Requires HMAC signature verification by default
                - "none": Applies no default authorization
            **kwargs: Additional arguments passed to FastAPI APIRouter.
        """
        self.router = APIRouter(prefix=prefix, tags=tags, **kwargs)
        self._app: Optional[FastAPI] = None
        self._extra_models: List[Any] = []
        self._auth_routes: List[str] = (
            []
        )  # Store paths of routes requiring authentication
        self._default_auth = default_auth  # Store default authorization strategy
        self._collect_routes()

    def _collect_routes(self):
        """
        Traverse all class members to find and register methods marked by route decorators.
        Apply the corresponding default authorization strategy based on the default_auth parameter.
        """
        for _member_name, member in inspect.getmembers(self):
            if callable(member) and hasattr(member, "__route_info__"):
                path, methods, route_kwargs = getattr(member, "__route_info__")

                # Collect extra_models
                extra_models = getattr(member, "__extra_models__", [])
                self._extra_models.extend(extra_models)

                # Apply default authorization (if no authorization decorator is present)
                authorized_member = self._apply_default_auth(member)

                # Check if authentication is required
                if self._needs_authentication(authorized_member):
                    # Record the path of routes requiring authentication, removing type annotations from path parameters
                    full_path = (
                        f"{self.router.prefix}{path}" if self.router.prefix else path
                    )
                    # Remove type annotations from path parameters, e.g., {resource_id:int} -> {resource_id}
                    clean_path = self._clean_path_types(full_path)
                    self._auth_routes.append(clean_path)

                self.router.add_api_route(
                    path, endpoint=authorized_member, methods=methods, **route_kwargs
                )

    def _apply_default_auth(self, func: Callable) -> Callable:
        """
        Apply default authorization strategy based on the default_auth parameter.

        Args:
            func: The function to check.

        Returns:
            Callable: The function with default authorization applied (if no authorization decorator exists).
        """
        # Check if the function already has an authorization decorator
        if hasattr(func, '__authorization_context__'):
            return func

        # If it's a bound method, get the original function
        if hasattr(func, '__func__'):
            # This is a bound method; check if the original function already has an authorization decorator
            if hasattr(func.__func__, '__authorization_context__'):
                return func

            # Get the original function and apply the decorator
            original_func = func.__func__
            decorated_func = self._get_auth_decorator()(original_func)
            # Rebind to the instance
            return decorated_func.__get__(func.__self__, type(func.__self__))
        else:
            # This is an unbound function; apply the decorator directly
            return self._get_auth_decorator()(func)

    def _get_auth_decorator(self):
        """Get the corresponding authorization decorator"""
        if self._default_auth == "require_user":
            from core.authorize.decorators import require_user

            return require_user
        elif self._default_auth == "require_anonymous":
            from core.authorize.decorators import require_anonymous

            return require_anonymous
        elif self._default_auth == "require_admin":
            from core.authorize.decorators import require_admin

            return require_admin
        elif self._default_auth == "require_signature":
            from core.authorize.decorators import require_signature

            return require_signature
        elif self._default_auth == "none":
            # Apply no default authorization; return an identity decorator
            return lambda x: x
        else:
            # Unknown authorization strategy; default to require_user
            from core.authorize.decorators import require_user

            return require_user

    def _needs_authentication(self, func: Callable) -> bool:
        """
        Check if the function requires authentication.

        Args:
            func: The function to check.

        Returns:
            bool: Whether authentication is required.
        """
        # Check if the function has authorization context directly
        if hasattr(func, '__authorization_context__'):
            auth_context = func.__authorization_context__
            return auth_context.need_auth()

        return False

    def _clean_path_types(self, path: str) -> str:
        """
        Clean type annotations from path parameters.

        Convert {resource_id:int} to {resource_id}
        Convert {user_id:str} to {user_id}

        Args:
            path: Path containing type annotations.

        Returns:
            str: Cleaned path.
        """
        import re

        # Use regular expression to match {parameter:type} format and replace with {parameter}
        return re.sub(r'\{([^}:]+):[^}]+\}', r'{\1}', path)

    def _get_security_config(self) -> List[dict]:
        """
        Get security configuration.

        Returns:
            List[dict]: List of security configurations.
        """
        # Prioritize class-level security configuration provider
        if self._security_config_provider is not None:
            return self._security_config_provider()

        # Try to get from global configuration
        try:
            from capabilities.auth.supabase_auth.supabase_auth_openapi import (
                get_security_config,
            )

            return get_security_config()
        except ImportError:
            return [{"OAuth2PasswordBearer": []}]

    def _is_union_type(self, model: Any) -> bool:
        """Check if it is a Union type"""
        return get_origin(model) is Union

    def _get_union_args(self, union_type: Any) -> tuple:
        """Get arguments of Union type"""
        return get_args(union_type)

    def _get_model_name(self, model: Any) -> str:
        """Get model name"""
        if hasattr(model, '__name__'):
            return model.__name__
        elif hasattr(model, '_name'):
            return model._name
        else:
            return str(model)

    def _generate_union_schema(self, union_type: Any, union_name: str) -> dict:
        """
        Generate oneOf schema structure for Union type.

        Args:
            union_type: Union type.
            union_name: Name of the Union type.

        Returns:
            Schema definition containing oneOf and discriminator.
        """
        union_args = self._get_union_args(union_type)

        # Generate oneOf array
        one_of = []
        discriminator_mapping = {}

        for arg in union_args:
            if hasattr(arg, '__name__'):
                model_name = arg.__name__
                one_of.append({"$ref": f"#/components/schemas/{model_name}"})

                # Try to get discriminator field value
                if hasattr(arg, 'model_fields') and 'type' in arg.model_fields:
                    # Get literal or enum value of the type field
                    type_field = arg.model_fields['type']
                    if (
                        hasattr(type_field, 'default')
                        and type_field.default is not None
                    ):
                        discriminator_mapping[type_field.default] = (
                            f"#/components/schemas/{model_name}"
                        )
                    elif hasattr(type_field.annotation, '__args__'):
                        # Handle Literal type
                        literal_values = getattr(type_field.annotation, '__args__', ())
                        if literal_values:
                            discriminator_mapping[literal_values[0]] = (
                                f"#/components/schemas/{model_name}"
                            )

        schema = {"oneOf": one_of}

        # Only add discriminator if there is a mapping
        if discriminator_mapping:
            schema["discriminator"] = {
                "propertyName": "type",
                "mapping": discriminator_mapping,
            }

        return schema

    def _custom_openapi_generator(self, app: FastAPI):
        """
        Custom OpenAPI generator to handle extra_models and authenticated routes.
        """

        def custom_openapi():
            if app.openapi_schema:
                return app.openapi_schema

            # Generate basic OpenAPI schema
            openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                summary=getattr(app, 'summary', None),
                description=app.description,
                routes=app.routes,
            )

            # Ensure components exist
            if "components" not in openapi_schema:
                openapi_schema["components"] = {}
            if "schemas" not in openapi_schema["components"]:
                openapi_schema["components"]["schemas"] = {}
            if "securitySchemes" not in openapi_schema["components"]:
                openapi_schema["components"]["securitySchemes"] = {}

            # Collect extra_models and authenticated routes from all BaseController instances
            controllers = []

            # Traverse all routes to find BaseController instances
            def collect_controllers_from_routes(routes):
                for route in routes:
                    if hasattr(route, 'router') and hasattr(route.router, 'routes'):
                        # This is an include_router case; process recursively
                        collect_controllers_from_routes(route.router.routes)
                    elif hasattr(route, 'endpoint') and hasattr(
                        route.endpoint, '__self__'
                    ):
                        # This is a bound method; check if it's a BaseController instance
                        controller = route.endpoint.__self__
                        if (
                            isinstance(controller, BaseController)
                            and controller not in controllers
                        ):
                            controllers.append(controller)

            collect_controllers_from_routes(app.routes)

            # Process extra_models for all controllers
            for controller in controllers:
                self._process_controller_extra_models(controller, openapi_schema)

            # Add security schemes definition to OpenAPI schema
            self._add_security_schemes_to_openapi(controllers, openapi_schema)

            # Add security configuration to routes requiring authentication
            self._add_security_to_auth_routes(controllers, openapi_schema)

            app.openapi_schema = openapi_schema
            return app.openapi_schema

        return custom_openapi

    def _add_security_schemes_to_openapi(
        self, controllers: List['BaseController'], openapi_schema: dict
    ):
        """
        Add security schemes definition to OpenAPI schema.

        Args:
            controllers: List of all BaseController instances.
            openapi_schema: OpenAPI schema dictionary.
        """
        # Collect security schemes used by all controllers
        security_schemes = {}

        for controller in controllers:
            # Check if the controller has a custom security configuration provider
            if (
                hasattr(controller, '_security_config_provider')
                and controller._security_config_provider is not None
            ):
                try:
                    # Try to get security schemes definition (if supported by controller)
                    if hasattr(controller, '_get_security_schemes'):
                        schemes = controller._get_security_schemes()
                        if schemes:
                            security_schemes.update(schemes)
                    else:
                        # Check if it's HMAC signature authentication
                        security_config = controller._security_config_provider()
                        if security_config and any(
                            "HMACSignature" in config for config in security_config
                        ):
                            # Import HMAC security schemes definition
                            try:
                                from core.middleware.hmac_signature_middleware import (
                                    get_hmac_openapi_security_schemes,
                                )

                                hmac_schemes = get_hmac_openapi_security_schemes()
                                security_schemes.update(hmac_schemes)
                            except ImportError:
                                # If import fails, use default HMAC definition
                                security_schemes["HMACSignature"] = {
                                    "type": "apiKey",
                                    "in": "header",
                                    "name": "X-Signature",
                                    "description": "HMAC signature authentication",
                                }
                except Exception as e:
                    # If getting security schemes fails, log error but don't affect other functionality
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to get security schemes definition for controller {controller.__class__.__name__}: {str(e)}"
                    )

        # Add security schemes definition to OpenAPI schema
        if security_schemes:
            openapi_schema["components"]["securitySchemes"].update(security_schemes)

    def _add_security_to_auth_routes(
        self, controllers: List['BaseController'], openapi_schema: dict
    ):
        """
        Add security configuration to routes requiring authentication.

        Args:
            controllers: List of all BaseController instances.
            openapi_schema: OpenAPI schema dictionary.
        """
        # Collect all paths requiring authentication
        all_auth_routes = []
        for controller in controllers:
            if hasattr(controller, '_auth_routes'):
                all_auth_routes.extend(controller._auth_routes)

        # Get security configuration
        security_config = self._get_security_config()

        # Add security configuration to routes requiring authentication
        if "paths" in openapi_schema:
            for path, path_item in openapi_schema["paths"].items():
                # Check if current path requires authentication
                if path in all_auth_routes:
                    # Add security configuration for all HTTP methods
                    for method in [
                        "get",
                        "post",
                        "put",
                        "delete",
                        "patch",
                        "head",
                        "options",
                    ]:
                        if method in path_item:
                            path_item[method]["security"] = security_config

    def _process_controller_extra_models(self, controller, openapi_schema):
        """
        Process extra_models for a single controller.
        """
        if not hasattr(controller, '_extra_models'):
            return

        for model in controller._extra_models:
            if self._is_union_type(model):
                # For Union types, we need to find their original names
                # Look up the variable name of this Union type in the controller's module
                model_name = None
                if hasattr(controller, '__class__') and hasattr(
                    controller.__class__, '__module__'
                ):
                    import sys

                    module = sys.modules.get(controller.__class__.__module__)
                    if module:
                        for attr_name in dir(module):
                            attr_value = getattr(module, attr_name)
                            if attr_value is model:
                                model_name = attr_name
                                break

                # If still not found, use default name
                if not model_name:
                    model_name = "Union"

                # Process Union type
                union_schema = self._generate_union_schema(model, model_name)
                openapi_schema["components"]["schemas"][model_name] = union_schema

                # Also add schemas for Union members
                union_args = self._get_union_args(model)
                for arg in union_args:
                    if hasattr(arg, 'model_json_schema'):
                        arg_name = self._get_model_name(arg)
                        if arg_name not in openapi_schema["components"]["schemas"]:
                            # Generate schema for individual model
                            arg_schema = arg.model_json_schema(
                                ref_template="#/components/schemas/{model}"
                            )
                            # Extract schemas from $defs
                            if '$defs' in arg_schema:
                                openapi_schema["components"]["schemas"].update(
                                    arg_schema['$defs']
                                )
                                del arg_schema['$defs']
                            # Add main model schema
                            openapi_schema["components"]["schemas"][
                                arg_name
                            ] = arg_schema
            else:
                # Process regular model
                model_name = self._get_model_name(model)
                if hasattr(model, 'model_json_schema'):
                    if model_name not in openapi_schema["components"]["schemas"]:
                        model_schema = model.model_json_schema(
                            ref_template="#/components/schemas/{model}"
                        )
                        # Extract schemas from $defs
                        if '$defs' in model_schema:
                            openapi_schema["components"]["schemas"].update(
                                model_schema['$defs']
                            )
                            del model_schema['$defs']
                        # Add main model schema
                        openapi_schema["components"]["schemas"][
                            model_name
                        ] = model_schema

    def register_to_app(self, app: FastAPI):
        """
        Register this controller's routes to the FastAPI application instance.

        Args:
            app (FastAPI): The main FastAPI application instance.
        """
        self._app = app
        app.include_router(self.router)

        # Reset custom OpenAPI generator each time a controller is registered
        # This ensures all controllers' extra_models are properly handled
        app.openapi = self._custom_openapi_generator(app)
        # Clear cached schema to force regeneration
        app.openapi_schema = None


# Usage examples:
#
# 1. Using default FastAPI Users authentication configuration:
# class UserController(BaseController):
#     def __init__(self):
#         super().__init__(prefix="/users", tags=["Users"])
#
#     @get("/")
#     @require_user  # Requires authentication
#     def list_users(self):
#         return [{"id": 1, "name": "user1"}]
#
# 2. Custom security configuration provider:
# class CustomAuthController(BaseController):
#     # Custom security configuration provider
#     _security_config_provider = lambda: [
#         {
#             "CustomAuth": []
#         }
#     ]
#
#     def __init__(self):
#         super().__init__(prefix="/custom", tags=["Custom"])
#
#     @get("/")
#     @require_user
#     def custom_endpoint(self):
#         return {"message": "Custom authenticated endpoint"}
#
# 3. Dynamic security configuration:
# class DynamicAuthController(BaseController):
#     def __init__(self, auth_type: str = "OAuth2PasswordBearer"):
#         super().__init__(prefix="/dynamic", tags=["Dynamic"])
#         self.auth_type = auth_type
#         # Dynamically set security configuration provider
#         self._security_config_provider = lambda: [
#             {
#                 self.auth_type: []
#             }
#         ]
