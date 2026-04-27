# -*- coding: utf-8 -*-
"""
Debug Controller

Provides debugging interfaces for Beans in the DI container, supports calling specific methods of specific services.
Only enabled in development environment, automatically disabled in production environment.
"""

import asyncio
import inspect
import os
import json
import traceback
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from fastapi import HTTPException

from core.interface.controller.base_controller import BaseController, post, get
from core.di import get_container, get_bean, get_bean_by_type
from core.di.decorators import controller
from core.observation.logger import get_logger

from core.constants.errors import ErrorMessage

logger = get_logger(__name__)


class BeanCallRequest(BaseModel):
    """Bean method call request model (compatible with code execution)"""

    # Bean identifier (choose one)
    bean_name: Optional[str] = Field(None, description="Bean name")
    bean_type: Optional[str] = Field(None, description="Bean type name")

    # Method call
    method: str = Field(..., description="Method name to be called")

    # Traditional parameter method
    args: List[Any] = Field(
        default_factory=list, description="List of positional arguments"
    )
    kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Dictionary of keyword arguments"
    )

    # Code execution method (optional)
    code: Optional[str] = Field(
        None,
        description="Python code used to generate args and kwargs parameters (optional)",
    )

    @classmethod
    def model_validate(cls, obj, *, strict=None, from_attributes=None, context=None):
        """Custom validation logic"""
        if (
            isinstance(obj, dict)
            and not obj.get("bean_name")
            and not obj.get("bean_type")
        ):
            logger.error(
                "Bean call request validation failed: missing bean_name or bean_type parameter"
            )
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)
        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )


class BeanCallWithCodeRequest(BaseModel):
    """Bean method call request model using code to generate parameters"""

    # Bean identifier (choose one)
    bean_name: Optional[str] = Field(None, description="Bean name")
    bean_type: Optional[str] = Field(None, description="Bean type name")

    # Method call
    method: str = Field(..., description="Method name to be called")

    # Python code to generate parameters
    code: str = Field(
        ..., description="Python code used to generate args and kwargs parameters"
    )

    @classmethod
    def model_validate(cls, obj, *, strict=None, from_attributes=None, context=None):
        """Custom validation logic"""
        if (
            isinstance(obj, dict)
            and not obj.get("bean_name")
            and not obj.get("bean_type")
        ):
            logger.error(
                "Bean code call request validation failed: missing bean_name or bean_type parameter"
            )
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)
        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )


class BeanCallResponse(BaseModel):
    """Bean method call response model"""

    success: bool = Field(..., description="Whether the call was successful")
    result: Optional[Any] = Field(
        None, description="Method return value (JSON serializable)"
    )
    result_str: Optional[str] = Field(
        None,
        description="String representation of the return value (used when not JSON serializable)",
    )
    error: Optional[str] = Field(None, description="Error message")
    traceback: Optional[str] = Field(None, description="Detailed error stack trace")
    bean_info: Optional[Dict[str, Any]] = Field(
        None, description="Information about the called Bean"
    )
    code_execution: Optional[Dict[str, Any]] = Field(
        None,
        description="Code execution information (only when using code to generate parameters)",
    )


class BeanInfoResponse(BaseModel):
    """Bean information response model"""

    name: str = Field(..., description="Bean name")
    type_name: str = Field(..., description="Bean type name")
    scope: str = Field(..., description="Bean scope")
    is_primary: bool = Field(..., description="Whether it is a Primary Bean")
    is_mock: bool = Field(..., description="Whether it is a Mock Bean")
    methods: List[str] = Field(
        default_factory=list, description="List of callable methods"
    )


@controller(name="debug_controller")
class DebugController(BaseController):
    """
    DI Container Debug API Controller

    Provides debugging and testing capabilities for the dependency injection container, supporting:
    - Viewing information of all registered Beans
    - Calling any method of any Bean for testing
    - Getting detailed configuration and method list of Beans
    - Monitoring the runtime status of the DI container

    **Security Mechanism**:
    - Only enabled in development environment (ENV=DEV)
    - All debugging interfaces are automatically disabled in production
    - No user authentication required, but access controlled by environment variables

    **Main Features**:
    1. **Bean Query**: Supports searching Beans by name or type
    2. **Method Call**: Supports passing parameters to call Bean methods
    3. **Status Monitoring**: View container mock mode, number of Beans, etc.
    4. **Error Diagnosis**: Provides detailed call error information and stack trace
    """

    def __init__(self):
        super().__init__(
            prefix="/asdf/debug/di",
            tags=["Debug"],
            default_auth="none",  # Debug interface does not require authentication, but access is controlled by environment variables
        )
        self.container = get_container()

    def _check_debug_enabled(self) -> bool:
        """Check if debugging is enabled"""
        return os.environ.get('ENV', 'prod').upper() == 'DEV'

    def _ensure_debug_enabled(self):
        """Ensure debugging is enabled, otherwise raise 404 error"""
        if not self._check_debug_enabled():
            logger.error("Debugging is not enabled, access to debug interface denied")
            raise HTTPException(
                status_code=404, detail=ErrorMessage.PERMISSION_DENIED.value
            )

    def _get_bean_methods(self, bean_instance: Any) -> List[str]:
        """Get list of callable methods of a Bean instance"""
        methods = []
        for attr_name in dir(bean_instance):
            if not attr_name.startswith('_'):  # Exclude private methods
                attr = getattr(bean_instance, attr_name)
                if callable(attr):
                    methods.append(attr_name)
        return sorted(methods)

    def _get_bean_by_identifier(
        self, bean_name: Optional[str], bean_type: Optional[str]
    ) -> tuple[Any, Dict[str, Any]]:
        """
        Get Bean instance and information by identifier

        Args:
            bean_name: Bean name
            bean_type: Bean type name

        Returns:
            tuple: (bean_instance, bean_info)

        Raises:
            HTTPException: When Bean is not found or parameters are invalid
        """
        if bean_name and bean_type:
            logger.error(
                "Bean identifier parameter error: cannot provide both bean_name and bean_type"
            )
            raise HTTPException(
                status_code=400, detail=ErrorMessage.INVALID_PARAMETER.value
            )

        if not bean_name and not bean_type:
            logger.error(
                "Bean identifier parameter error: must provide either bean_name or bean_type"
            )
            raise HTTPException(
                status_code=400, detail=ErrorMessage.INVALID_PARAMETER.value
            )

        try:
            if bean_name:
                # Get Bean by name
                bean_instance = get_bean(bean_name)
                bean_info = {
                    "name": bean_name,
                    "type_name": type(bean_instance).__name__,
                    "lookup_method": "by_name",
                }
            else:
                # Get Bean by type name
                # First need to find the corresponding type
                bean_class = self._find_bean_type_by_name(bean_type)
                if not bean_class:
                    logger.error(f"Bean class with type '{bean_type}' not found")
                    raise HTTPException(
                        status_code=404, detail=ErrorMessage.BEAN_NOT_FOUND.value
                    )

                bean_instance = get_bean_by_type(bean_class)
                bean_info = {
                    "name": getattr(bean_instance, '_di_name', bean_type.lower()),
                    "type_name": bean_type,
                    "lookup_method": "by_type",
                }

            return bean_instance, bean_info

        except Exception as e:
            if "not found" in str(e).lower():
                identifier = bean_name or bean_type
                method = "name" if bean_name else "type"
                logger.error(f"Bean not found by {method} '{identifier}': {str(e)}")
                raise HTTPException(
                    status_code=404, detail=ErrorMessage.BEAN_NOT_FOUND.value
                ) from e
            else:
                logger.error(f"Error occurred while getting Bean: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
                ) from e

    def _find_bean_type_by_name(self, type_name: str) -> Optional[Type]:
        """
        Find the corresponding Bean type by type name

        Uses a more reliable approach: get all Beans and check their types,
        avoiding reliance on potentially inaccurate list_all_beans_info

        Args:
            type_name: Type name

        Returns:
            Corresponding type, or None if not found
        """
        try:
            # Method 1: Use get_beans() to get all Bean instances, check type names
            all_beans_dict = self.container.get_beans()

            for _, bean_instance in all_beans_dict.items():
                if bean_instance is not None:
                    bean_type = type(bean_instance)
                    if bean_type.__name__ == type_name:
                        return bean_type

            return None

        except Exception:
            # If get_beans() fails, use fallback method
            # Try to infer using common type name patterns
            try:
                # First try to get all Bean info as fallback
                all_beans = self.container.list_all_beans_info()

                for bean_info in all_beans:
                    if bean_info['type_name'] == type_name:
                        # Get Bean instance by name, then get its type
                        try:
                            bean_instance = get_bean(bean_info['name'])
                            return type(bean_instance)
                        except Exception:
                            continue

                return None

            except Exception:
                # If all methods fail, return None
                return None

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """
        Serialize method call result

        Args:
            result: Method return value

        Returns:
            Dictionary containing serialized result
        """
        try:
            # Try JSON serialization
            json.dumps(result)
            return {"result": result}
        except (TypeError, ValueError):
            # If JSON serialization fails, return string representation
            return {"result_str": repr(result)}

    def _execute_parameter_code(self, code: str) -> Dict[str, Any]:
        """
        Safely execute Python code to generate parameters

        Args:
            code: Python code string

        Returns:
            Dictionary containing args and kwargs

        Raises:
            ValueError: When code execution fails or return format is incorrect
        """
        try:
            # Create safe execution environment, allow free imports
            safe_globals = {
                '__builtins__': {
                    # Basic types and functions
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'print': print,
                    'isinstance': isinstance,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                    'type': type,
                    'abs': abs,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'sorted': sorted,
                    'reversed': reversed,
                    'any': any,
                    'all': all,
                    'map': map,
                    'filter': filter,
                    # Allow import
                    '__import__': __import__,
                },
                # Pre-import commonly used modules and types
                'datetime': None,
                'json': None,
                'uuid': None,
                'typing': None,
            }

            # Pre-import commonly used modules
            try:
                import datetime
                import json
                import uuid
                import typing

                safe_globals['datetime'] = datetime
                safe_globals['json'] = json
                safe_globals['uuid'] = uuid
                safe_globals['typing'] = typing
            except ImportError as e:
                logger.warning(f"Failed to pre-import module: {e}")

            # No longer pre-import internal project modules, support completely free imports

            local_vars = {}

            # Execute code
            exec(code, safe_globals, local_vars)

            # Check if args and kwargs are defined
            if 'args' not in local_vars and 'kwargs' not in local_vars:
                logger.error(
                    "Invalid code execution result: args or kwargs variable not defined"
                )
                raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

            args = local_vars.get('args', [])
            kwargs = local_vars.get('kwargs', {})

            # Validate types
            if not isinstance(args, (list, tuple)):
                logger.error(
                    f"Wrong args parameter type: expected list or tuple, got {type(args).__name__}"
                )
                raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

            if not isinstance(kwargs, dict):
                logger.error(
                    f"Wrong kwargs parameter type: expected dict, got {type(kwargs).__name__}"
                )
                raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

            return {'args': list(args), 'kwargs': kwargs}

        except Exception as e:
            logger.error(f"Failed to execute parameter generation code: {e}")
            logger.error(f"Code execution exception details: {str(e)}")
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

    @get(
        "/status",
        response_model=Dict[str, Any],
        summary="Get debugging function status",
        responses={
            200: {
                "description": "Debugging status information retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "debug_enabled": True,
                            "container_info": {"mock_mode": False, "total_beans": 15},
                        }
                    }
                },
            }
        },
    )
    def get_debug_status(self) -> Dict[str, Any]:
        """
        Get debugging function status information

        Returns whether debugging is enabled and basic information about the DI container:
        - debug_enabled: Whether debugging is enabled (based on ENV environment variable)
        - container_info: DI container information, including mock mode status and total number of Beans

        **Note**:
        - debug_enabled is false when ENV != 'DEV'
        - This interface is not controlled by the debug switch and is always accessible

        Returns:
            Dict[str, Any]: Dictionary containing debugging status and container information
        """
        return {
            "debug_enabled": self._check_debug_enabled(),
            "container_info": {
                "mock_mode": self.container.is_mock_mode(),
                "total_beans": len(self.container.list_all_beans_info()),
            },
        }

    @get(
        "/beans",
        extra_models=[BeanInfoResponse],
        response_model=List[BeanInfoResponse],
        summary="List all registered Bean information",
        responses={
            200: {
                "description": "Bean list retrieved successfully",
                "content": {
                    "application/json": {
                        "example": [
                            {
                                "name": "user_service",
                                "type_name": "UserService",
                                "scope": "singleton",
                                "is_primary": True,
                                "is_mock": False,
                                "methods": [
                                    "get_user",
                                    "create_user",
                                    "update_user",
                                    "delete_user",
                                ],
                            },
                            {
                                "name": "email_service",
                                "type_name": "EmailService",
                                "scope": "singleton",
                                "is_primary": False,
                                "is_mock": False,
                                "methods": ["send_email", "validate_email"],
                            },
                        ]
                    }
                },
            },
            404: {"description": "Debugging not enabled or Bean not found"},
            500: {"description": "Internal error occurred while retrieving Bean list"},
        },
    )
    def list_all_beans(self) -> List[BeanInfoResponse]:
        """
        List all registered Bean information

        Returns a detailed list of all registered Beans in the DI container, including:
        - name: Bean name
        - type_name: Bean type name
        - scope: Bean scope (singleton/prototype/factory)
        - is_primary: Whether it is a Primary Bean
        - is_mock: Whether it is a Mock Bean
        - methods: List of callable public methods

        **Note**:
        - Only available when debugging mode is enabled (ENV=DEV)
        - Method list only includes public methods not starting with underscore
        - If getting method list for a Bean fails, the Bean is still returned but methods is empty

        Returns:
            List[BeanInfoResponse]: List of Bean information

        Raises:
            HTTPException: When debugging is not enabled or retrieving Bean list fails
        """
        self._ensure_debug_enabled()

        try:
            beans_info = []
            all_beans = self.container.list_all_beans_info()

            for bean_info in all_beans:
                try:
                    # Get Bean instance to retrieve method list
                    bean_instance = get_bean(bean_info['name'])
                    methods = self._get_bean_methods(bean_instance)

                    beans_info.append(
                        BeanInfoResponse(
                            name=bean_info['name'],
                            type_name=bean_info['type_name'],
                            scope=bean_info['scope'],
                            is_primary=bean_info['is_primary'],
                            is_mock=bean_info['is_mock'],
                            methods=methods,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to get method list for Bean '%s': %s",
                        bean_info['name'],
                        str(e),
                    )
                    # Even if getting method list fails, return basic information
                    beans_info.append(
                        BeanInfoResponse(
                            name=bean_info['name'],
                            type_name=bean_info['type_name'],
                            scope=bean_info['scope'],
                            is_primary=bean_info['is_primary'],
                            is_mock=bean_info['is_mock'],
                            methods=[],
                        )
                    )

            return beans_info

        except Exception as e:
            logger.error("Error occurred while listing all Beans: %s", str(e))
            logger.error(f"Exception details when retrieving Bean list: {str(e)}")
            raise HTTPException(
                status_code=500, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
            ) from e

    @post(
        "/call",
        extra_models=[BeanCallRequest, BeanCallResponse],
        response_model=BeanCallResponse,
        summary="Call specified method of specified Bean",
        responses={
            200: {
                "description": "Bean method call succeeded",
                "content": {
                    "application/json": {
                        "examples": {
                            "traditional_way": {
                                "summary": "Traditional way call succeeded",
                                "value": {
                                    "success": True,
                                    "result": ["uuid1", "uuid2", "uuid3"],
                                    "bean_info": {
                                        "name": "resource_repository",
                                        "type_name": "SQLModelResourceRepositoryImpl",
                                        "lookup_method": "by_name",
                                    },
                                },
                            },
                            "code_execution_way": {
                                "summary": "Code execution way call succeeded",
                                "value": {
                                    "success": True,
                                    "result": [
                                        "d7a8782f-d35f-48fb-81fb-ce2fa3c01cdf",
                                        "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                                    ],
                                    "bean_info": {
                                        "name": "resource_repository",
                                        "type_name": "SQLModelResourceRepositoryImpl",
                                        "lookup_method": "by_name",
                                    },
                                    "code_execution": {
                                        "generated_args": [],
                                        "generated_kwargs": {
                                            "resource_ids": [274, 281, 282],
                                            "resource_type": "LITERATURE",
                                            "user_id": 1,
                                        },
                                    },
                                },
                            },
                            "failure": {
                                "summary": "Call failed, includes error information",
                                "value": {
                                    "success": False,
                                    "error": "Bean not found",
                                    "traceback": "Traceback (most recent call last):\n  File ...",
                                    "bean_info": None,
                                },
                            },
                        }
                    }
                },
            },
            400: {
                "description": "Request parameter error, such as missing required parameters, invalid Bean identifier, etc."
            },
            404: {
                "description": "Debugging not enabled, Bean does not exist, or method does not exist"
            },
            500: {"description": "Internal error occurred during method call"},
        },
    )
    async def call_bean_method(self, request: BeanCallRequest) -> BeanCallResponse:
        """
        Call specified method of specified Bean (compatible with code execution)

        Supports two parameter passing methods:
        1. **Traditional method**: Directly use `args` and `kwargs` parameters
        2. **Code execution**: Use `code` parameter to dynamically generate parameters (supports enum types and complex objects)

        **Bean identification methods**:
        - bean_name/bean_type: Choose one, used to identify the Bean to be called

        **Parameter passing methods**:
        - Traditional method: Use `args` and `kwargs` fields
        - Code execution: Use `code` field to write Python code to generate parameters

        **Code execution example**:
        ```json
        {
            "bean_name": "resource_repository",
            "method": "get_uuids_by_ids_and_type",
            "code": "from domain.models.enums import ResourceType\\n\\nargs = []\\nkwargs = {\\n    'resource_ids': [274, 281, 282],\\n    'resource_type': ResourceType.LITERATURE,\\n    'user_id': 1\\n}"
        }
        ```

        **Traditional method example**:
        ```json
        {
            "bean_name": "resource_repository",
            "method": "get_by_ids",
            "args": [],
            "kwargs": {
                "resource_ids": [274, 281, 282]
            }
        }
        ```

        Args:
            request: Bean method call request

        Returns:
            BeanCallResponse: Method call result

        Raises:
            HTTPException: When debugging is not enabled, parameters are invalid, Bean does not exist, or method call fails
        """
        self._ensure_debug_enabled()

        try:
            # Determine which parameter method to use
            if request.code:
                # Use code execution method
                logger.info("Using code execution method to generate parameters")
                code_result = self._execute_parameter_code(request.code)
                args = code_result['args']
                kwargs = code_result['kwargs']
                code_execution_info = {
                    'generated_args': args,
                    'generated_kwargs': kwargs,
                }
            else:
                # Use traditional method
                logger.info("Using traditional parameter method")
                args = request.args
                kwargs = request.kwargs
                code_execution_info = None

            # Get Bean instance and information
            bean_instance, bean_info = self._get_bean_by_identifier(
                request.bean_name, request.bean_type
            )

            # Check if method exists
            if not hasattr(bean_instance, request.method):
                logger.error(
                    f"Method '{request.method}' does not exist in Bean '{bean_info['name']}'"
                )
                raise HTTPException(
                    status_code=404, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
                )

            method_to_call = getattr(bean_instance, request.method)

            # Check if it is a callable object
            if not callable(method_to_call):
                logger.error(
                    f"Attribute '{request.method}' of Bean '{bean_info['name']}' is not callable"
                )
                raise HTTPException(
                    status_code=400, detail=ErrorMessage.INVALID_PARAMETER.value
                )

            # Call method
            logger.info(
                "Calling Bean method: %s.%s(args=%s, kwargs=%s)",
                bean_info['name'],
                request.method,
                args,
                kwargs,
            )

            # Check if it is a coroutine function, compatible with async and sync methods
            if asyncio.iscoroutinefunction(
                method_to_call
            ) or inspect.iscoroutinefunction(method_to_call):
                result = await method_to_call(*args, **kwargs)
            else:
                result = method_to_call(*args, **kwargs)

            # Serialize result
            serialized_result = self._serialize_result(result)

            # Construct response
            response_data = {
                'success': True,
                'bean_info': bean_info,
                **serialized_result,
            }

            # If code execution was used, add execution information
            if code_execution_info:
                response_data['code_execution'] = code_execution_info

            return BeanCallResponse(**response_data)

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Catch and handle other exceptions
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            logger.error("Error occurred while calling Bean method: %s", error_msg)
            logger.debug("Error stack: %s", error_traceback)

            return BeanCallResponse(
                success=False,
                error=error_msg,
                traceback=error_traceback,
                bean_info=getattr(locals(), 'bean_info', None),
                code_execution=getattr(locals(), 'code_execution_info', None),
            )

    @get(
        "/beans/{bean_name}",
        extra_models=[BeanInfoResponse],
        response_model=BeanInfoResponse,
        summary="Get detailed information by Bean name",
        responses={
            200: {
                "description": "Bean information retrieved successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "name": "user_service",
                            "type_name": "UserService",
                            "scope": "singleton",
                            "is_primary": True,
                            "is_mock": False,
                            "methods": [
                                "get_user",
                                "create_user",
                                "update_user",
                                "delete_user",
                                "list_users",
                                "validate_user",
                            ],
                        }
                    }
                },
            },
            404: {
                "description": "Debugging not enabled or specified Bean does not exist"
            },
            500: {
                "description": "Internal error occurred while retrieving Bean information"
            },
        },
    )
    def get_bean_info(self, bean_name: str) -> BeanInfoResponse:
        """
        Get detailed information by Bean name

        Query complete information of a specific Bean by its name, including type, scope,
        whether it is a Primary Bean, whether it is a Mock Bean, and a list of all callable public methods.

        **Returned information includes**:
        - name: Bean name
        - type_name: Bean type name
        - scope: Bean scope (singleton/prototype/factory)
        - is_primary: Whether it is a Primary Bean (preferred Bean when multiple Beans of the same type exist)
        - is_mock: Whether it is a Mock Bean (mock implementation used in test environments)
        - methods: List of callable public methods (excluding private methods starting with underscore)

        **Use cases**:
        - View detailed configuration information of a specific Bean
        - Understand all callable methods provided by a Bean
        - Debug registration status of Beans in the DI container

        **Notes**:
        - Only available when debugging mode is enabled (ENV=DEV)
        - Bean name must match exactly, case-sensitive
        - Method list is sorted alphabetically

        Args:
            bean_name: Bean registration name, must exactly match the name in the DI container

        Returns:
            BeanInfoResponse: Detailed information of the Bean, including metadata and method list

        Raises:
            HTTPException: When debugging is not enabled, Bean does not exist, or retrieving information fails
        """
        self._ensure_debug_enabled()

        try:
            bean_instance, _ = self._get_bean_by_identifier(bean_name, None)
            methods = self._get_bean_methods(bean_instance)

            # Get Bean metadata from container
            all_beans = self.container.list_all_beans_info()
            bean_meta = next((b for b in all_beans if b['name'] == bean_name), None)

            if not bean_meta:
                raise HTTPException(
                    status_code=404, detail=ErrorMessage.BEAN_NOT_FOUND.value
                )

            return BeanInfoResponse(
                name=bean_meta['name'],
                type_name=bean_meta['type_name'],
                scope=bean_meta['scope'],
                is_primary=bean_meta['is_primary'],
                is_mock=bean_meta['is_mock'],
                methods=methods,
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error occurred while retrieving Bean information: %s", str(e))
            raise HTTPException(
                status_code=500, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
            ) from e

    @post(
        "/call-with-code",
        extra_models=[BeanCallWithCodeRequest, BeanCallResponse],
        response_model=BeanCallResponse,
        summary="Call Bean method by generating parameters with Python code",
        responses={
            200: {
                "description": "Bean method call succeeded",
                "content": {
                    "application/json": {
                        "examples": {
                            "success_with_enum": {
                                "summary": "Call succeeded with enum parameter",
                                "value": {
                                    "success": True,
                                    "result": ["uuid1", "uuid2", "uuid3"],
                                    "bean_info": {
                                        "name": "resource_repository",
                                        "type_name": "SQLModelResourceRepositoryImpl",
                                        "lookup_method": "by_name",
                                    },
                                    "code_execution": {
                                        "generated_args": [],
                                        "generated_kwargs": {
                                            "resource_ids": [274, 281, 282],
                                            "resource_type": "LITERATURE",
                                            "user_id": 1,
                                        },
                                    },
                                },
                            }
                        }
                    }
                },
            },
            400: {"description": "Request parameter error or code execution failed"},
            404: {
                "description": "Debugging not enabled, Bean does not exist, or method does not exist"
            },
            500: {"description": "Internal error occurred during method call"},
        },
    )
    async def call_bean_method_with_code(
        self, request: BeanCallWithCodeRequest
    ) -> BeanCallResponse:
        """
        Call Bean method by generating parameters with Python code

        This interface allows you to write Python code to dynamically generate method parameters, especially suitable for:
        1. **Enum type parameters**: e.g., ResourceType.LITERATURE
        2. **Complex object construction**: e.g., AIInputValueObject instance
        3. **Dynamic parameter calculation**: Generate parameter values based on logic
        4. **Type conversion**: Handle Python types that cannot be directly represented in JSON

        **Code execution environment**:
        - Provides a safe execution environment, limiting available built-in functions
        - Automatically imports common enum types: ResourceType, ResourceScope, ResourceProcessingStatus
        - Automatically imports common value objects: AIInputValueObject
        - Code must define `args` and/or `kwargs` variables

        **Code examples**:
        ```python
        # Example 1: Using enum type
        args = []
        kwargs = {
            'resource_ids': [274, 281, 282],
            'resource_type': ResourceType.LITERATURE,
            'user_id': 1
        }

        # Example 2: Constructing complex object
        ai_input = AIInputValueObject({
            'literature_refs': [
                {'value': {'id': 280}},
                {'value': {'id': 'uuid-string'}}
            ]
        })
        args = [ai_input]
        kwargs = {'user_id': 1}

        # Example 3: Dynamically calculating parameters
        resource_ids = list(range(270, 285))  # Generate ID list
        kwargs = {
            'resource_ids': resource_ids,
            'resource_type': ResourceType.LITERATURE,
            'user_id': 1
        }
        ```

        **Security restrictions**:
        - File operations, network access, and other dangerous functions are disabled
        - Only predefined safe functions and imported types can be used
        - Code execution timeout protection

        Args:
            request: Request containing Bean identifier, method name, and Python code

        Returns:
            BeanCallResponse: Method call result, including code execution information

        Raises:
            HTTPException: When debugging is not enabled, code execution fails, or method call fails
        """
        self._ensure_debug_enabled()

        try:
            # Execute code to generate parameters
            logger.info(
                "Executing parameter generation code: %s",
                request.code[:100] + "..." if len(request.code) > 100 else request.code,
            )

            code_result = self._execute_parameter_code(request.code)
            args = code_result['args']
            kwargs = code_result['kwargs']

            logger.info(
                "Code execution succeeded, generated parameters: args=%s, kwargs=%s",
                args,
                kwargs,
            )

            # Get Bean instance and information
            bean_instance, bean_info = self._get_bean_by_identifier(
                request.bean_name, request.bean_type
            )

            # Check if method exists
            if not hasattr(bean_instance, request.method):
                raise HTTPException(
                    status_code=404, detail=ErrorMessage.BEAN_OPERATION_FAILED.value
                )

            method_to_call = getattr(bean_instance, request.method)

            # Check if it is a callable object
            if not callable(method_to_call):
                raise HTTPException(
                    status_code=400, detail=ErrorMessage.INVALID_PARAMETER.value
                )

            # Call method
            logger.info(
                "Calling Bean method: %s.%s(args=%s, kwargs=%s)",
                bean_info['name'],
                request.method,
                args,
                kwargs,
            )

            # Check if it is a coroutine function, compatible with async and sync methods
            if asyncio.iscoroutinefunction(
                method_to_call
            ) or inspect.iscoroutinefunction(method_to_call):
                result = await method_to_call(*args, **kwargs)
            else:
                result = method_to_call(*args, **kwargs)

            # Serialize result
            serialized_result = self._serialize_result(result)

            # Add code execution information
            response_data = {
                'success': True,
                'bean_info': bean_info,
                'code_execution': {'generated_args': args, 'generated_kwargs': kwargs},
                **serialized_result,
            }

            return BeanCallResponse(**response_data)

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Catch and handle other exceptions
            error_msg = str(e)
            error_traceback = traceback.format_exc()

            logger.error(
                "Error occurred while calling Bean method with code: %s", error_msg
            )
            logger.debug("Error stack: %s", error_traceback)

            return BeanCallResponse(
                success=False,
                error=error_msg,
                traceback=error_traceback,
                bean_info=getattr(locals(), 'bean_info', None),
                code_execution=getattr(locals(), 'code_result', None),
            )
