import functools
import json
from typing import Any

from mcp.types import Tool as MCPTool

from .. import _debug
from ..exceptions import AgentsException, ModelBehaviorError, UserError
from ..logger import logger
from ..run_context import RunContextWrapper
from ..tool import FunctionTool, Tool
from .server import MCPServer


class MCPUtil:
    """Set of utilities for interop between MCP and Agents SDK tools."""

    @classmethod
    async def get_all_function_tools(cls, servers: list[MCPServer]) -> list[Tool]:
        """Get all function tools from a list of MCP servers."""
        tools = []
        tool_names: set[str] = set()
        for server in servers:
            server_tools = await cls.get_function_tools(server)
            server_tool_names = {tool.name for tool in server_tools}
            if len(server_tool_names & tool_names) > 0:
                raise UserError(
                    f"Duplicate tool names found across MCP servers: "
                    f"{server_tool_names & tool_names}"
                )
            tool_names.update(server_tool_names)
            tools.extend(server_tools)

        return tools

    @classmethod
    async def get_function_tools(cls, server: MCPServer) -> list[Tool]:
        """Get all function tools from a single MCP server."""
        tools = await server.list_tools()
        return [cls.to_function_tool(tool, server) for tool in tools]

    @classmethod
    def to_function_tool(cls, tool: MCPTool, server: MCPServer) -> FunctionTool:
        """Convert an MCP tool to an Agents SDK function tool."""
        invoke_func = functools.partial(cls.invoke_mcp_tool, server, tool)
        return FunctionTool(
            name=tool.name,
            description=tool.description or "",
            params_json_schema=tool.inputSchema,
            on_invoke_tool=invoke_func,
            strict_json_schema=False,
        )

    @classmethod
    async def invoke_mcp_tool(
        cls, server: MCPServer, tool: MCPTool, context: RunContextWrapper[Any], input_json: str
    ) -> str:
        """Invoke an MCP tool and return the result as a string."""
        try:
            json_data: dict[str, Any] = json.loads(input_json) if input_json else {}
        except Exception as e:
            if _debug.DONT_LOG_TOOL_DATA:
                logger.debug(f"Invalid JSON input for tool {tool.name}")
            else:
                logger.debug(f"Invalid JSON input for tool {tool.name}: {input_json}")
            raise ModelBehaviorError(
                f"Invalid JSON input for tool {tool.name}: {input_json}"
            ) from e

        if _debug.DONT_LOG_TOOL_DATA:
            logger.debug(f"Invoking MCP tool {tool.name}")
        else:
            logger.debug(f"Invoking MCP tool {tool.name} with input {input_json}")

        try:
            result = await server.call_tool(tool.name, json_data)
        except Exception as e:
            logger.error(f"Error invoking MCP tool {tool.name}: {e}")
            raise AgentsException(f"Error invoking MCP tool {tool.name}: {e}") from e

        if _debug.DONT_LOG_TOOL_DATA:
            logger.debug(f"MCP tool {tool.name} completed.")
        else:
            logger.debug(f"MCP tool {tool.name} returned {result}")

        # The MCP tool result is a list of content items, whereas OpenAI tool outputs are a single
        # string. We'll try to convert.
        if len(result.content) == 1:
            return result.content[0].model_dump_json()
        elif len(result.content) > 1:
            return json.dumps([item.model_dump() for item in result.content])
        else:
            logger.error(f"Errored MCP tool result: {result}")
            return "Error running tool."
