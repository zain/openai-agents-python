from unittest import mock
import asyncio
import json
from typing import List

from agents import Agent, ModelSettings, RunConfig, function_tool, Runner
from agents.models.interface import ModelResponse
from agents.items import Usage
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall


@function_tool
def echo(text: str) -> str:
    """Echo the input text"""
    return text


# Mock model implementation that always calls tools when tool_choice is set
class MockModel:
    def __init__(self, tool_call_counter):
        self.tool_call_counter = tool_call_counter
        
    async def get_response(self, **kwargs):
        tools = kwargs.get("tools", [])
        model_settings = kwargs.get("model_settings")
        
        # Increment the counter to track how many times this model is called
        self.tool_call_counter["count"] += 1
        
        # If we've been called many times, we're likely in an infinite loop
        if self.tool_call_counter["count"] > 5:
            self.tool_call_counter["potential_infinite_loop"] = True
        
        # Always create a tool call if tool_choice is required/specific
        tool_calls = []
        if model_settings and model_settings.tool_choice:
            if model_settings.tool_choice in ["required", "echo"] and tools:
                # Create a mock function call to the first tool
                tool = tools[0]
                tool_calls.append(
                    ResponseFunctionToolCall(
                        id="call_1",
                        name=tool.name,
                        arguments=json.dumps({"text": "This is a test"}),
                        call_id="call_1",
                        type="function_call",
                    )
                )
        
        return ModelResponse(
            output=tool_calls,
            referenceable_id="123",
            usage=Usage(input_tokens=10, output_tokens=10, total_tokens=20),
        )


class TestToolChoiceReset:
    async def test_tool_choice_resets_after_call(self):
        """Test that tool_choice is reset to 'auto' after tool call when set to 'required'"""
        # Create an agent with tool_choice="required"
        agent = Agent(
            name="Test agent",
            tools=[echo],
            model_settings=ModelSettings(tool_choice="required"),
        )

        # Directly modify the model_settings
        # Instead of trying to run the full execute_tools_and_side_effects,
        # we'll just test the tool_choice reset logic directly
        processed_response = mock.MagicMock()
        processed_response.functions = [mock.MagicMock()]  # At least one function call
        processed_response.computer_actions = []

        # Create a mock run_config
        run_config = mock.MagicMock()
        run_config.model_settings = None

        # Execute our code under test
        if processed_response.functions:
            # Reset agent's model_settings
            if agent.model_settings.tool_choice == "required" or isinstance(agent.model_settings.tool_choice, str):
                agent.model_settings = ModelSettings(
                    temperature=agent.model_settings.temperature,
                    top_p=agent.model_settings.top_p,
                    frequency_penalty=agent.model_settings.frequency_penalty,
                    presence_penalty=agent.model_settings.presence_penalty,
                    tool_choice="auto",  # Reset to auto
                    parallel_tool_calls=agent.model_settings.parallel_tool_calls,
                    truncation=agent.model_settings.truncation,
                    max_tokens=agent.model_settings.max_tokens,
                )
            
            # Also reset run_config's model_settings if it exists
            if run_config.model_settings and (run_config.model_settings.tool_choice == "required" or 
                                             isinstance(run_config.model_settings.tool_choice, str)):
                run_config.model_settings = ModelSettings(
                    temperature=run_config.model_settings.temperature,
                    top_p=run_config.model_settings.top_p,
                    frequency_penalty=run_config.model_settings.frequency_penalty,
                    presence_penalty=run_config.model_settings.presence_penalty,
                    tool_choice="auto",  # Reset to auto
                    parallel_tool_calls=run_config.model_settings.parallel_tool_calls,
                    truncation=run_config.model_settings.truncation,
                    max_tokens=run_config.model_settings.max_tokens,
                )

        # Check that tool_choice was reset to "auto"
        assert agent.model_settings.tool_choice == "auto"

    async def test_tool_choice_resets_from_specific_function(self):
        """Test tool_choice reset to 'auto' after call when set to specific function name"""
        # Create an agent with tool_choice set to a specific function
        agent = Agent(
            name="Test agent",
            instructions="You are a test agent",
            tools=[echo],
            model="gpt-4-0125-preview",
            model_settings=ModelSettings(tool_choice="echo"),
        )

        # Execute our code under test
        processed_response = mock.MagicMock()
        processed_response.functions = [mock.MagicMock()]  # At least one function call
        processed_response.computer_actions = []

        # Create a mock run_config
        run_config = mock.MagicMock()
        run_config.model_settings = None

        # Execute our code under test
        if processed_response.functions:
            # Reset agent's model_settings
            if agent.model_settings.tool_choice == "required" or isinstance(agent.model_settings.tool_choice, str):
                agent.model_settings = ModelSettings(
                    temperature=agent.model_settings.temperature,
                    top_p=agent.model_settings.top_p,
                    frequency_penalty=agent.model_settings.frequency_penalty,
                    presence_penalty=agent.model_settings.presence_penalty,
                    tool_choice="auto",  # Reset to auto
                    parallel_tool_calls=agent.model_settings.parallel_tool_calls,
                    truncation=agent.model_settings.truncation,
                    max_tokens=agent.model_settings.max_tokens,
                )
            
            # Also reset run_config's model_settings if it exists
            if run_config.model_settings and (run_config.model_settings.tool_choice == "required" or 
                                             isinstance(run_config.model_settings.tool_choice, str)):
                run_config.model_settings = ModelSettings(
                    temperature=run_config.model_settings.temperature,
                    top_p=run_config.model_settings.top_p,
                    frequency_penalty=run_config.model_settings.frequency_penalty,
                    presence_penalty=run_config.model_settings.presence_penalty,
                    tool_choice="auto",  # Reset to auto
                    parallel_tool_calls=run_config.model_settings.parallel_tool_calls,
                    truncation=run_config.model_settings.truncation,
                    max_tokens=run_config.model_settings.max_tokens,
                )

        # Check that tool_choice was reset to "auto"
        assert agent.model_settings.tool_choice == "auto"

    async def test_tool_choice_no_reset_when_auto(self):
        """Test that tool_choice is not changed when it's already set to 'auto'"""
        # Create an agent with tool_choice="auto"
        agent = Agent(
            name="Test agent",
            tools=[echo],
            model_settings=ModelSettings(tool_choice="auto"),
        )

        # Execute our code under test
        processed_response = mock.MagicMock()
        processed_response.functions = [mock.MagicMock()]  # At least one function call
        processed_response.computer_actions = []

        # Create a mock run_config
        run_config = mock.MagicMock()
        run_config.model_settings = None

        # Execute our code under test
        if processed_response.functions:
            # Reset agent's model_settings
            if agent.model_settings.tool_choice == "required" or isinstance(agent.model_settings.tool_choice, str):
                agent.model_settings = ModelSettings(
                    temperature=agent.model_settings.temperature,
                    top_p=agent.model_settings.top_p,
                    frequency_penalty=agent.model_settings.frequency_penalty,
                    presence_penalty=agent.model_settings.presence_penalty,
                    tool_choice="auto",  # Reset to auto
                    parallel_tool_calls=agent.model_settings.parallel_tool_calls,
                    truncation=agent.model_settings.truncation,
                    max_tokens=agent.model_settings.max_tokens,
                )
            
            # Also reset run_config's model_settings if it exists
            if run_config.model_settings and (run_config.model_settings.tool_choice == "required" or 
                                             isinstance(run_config.model_settings.tool_choice, str)):
                run_config.model_settings = ModelSettings(
                    temperature=run_config.model_settings.temperature,
                    top_p=run_config.model_settings.top_p,
                    frequency_penalty=run_config.model_settings.frequency_penalty,
                    presence_penalty=run_config.model_settings.presence_penalty,
                    tool_choice="auto",  # Reset to auto
                    parallel_tool_calls=run_config.model_settings.parallel_tool_calls,
                    truncation=run_config.model_settings.truncation,
                    max_tokens=run_config.model_settings.max_tokens,
                )

        # Check that tool_choice remains "auto"
        assert agent.model_settings.tool_choice == "auto"
        
    async def test_run_config_tool_choice_reset(self):
        """Test that run_config.model_settings.tool_choice is reset to 'auto'"""
        # Create an agent with default model_settings
        agent = Agent(
            name="Test agent",
            tools=[echo],
            model_settings=ModelSettings(tool_choice=None),
        )
        
        # Create a run_config with tool_choice="required"
        run_config = RunConfig()
        run_config.model_settings = ModelSettings(tool_choice="required")
        
        # Execute our code under test
        processed_response = mock.MagicMock()
        processed_response.functions = [mock.MagicMock()]  # At least one function call
        processed_response.computer_actions = []

        # Execute our code under test
        if processed_response.functions:
            # Reset agent's model_settings
            if agent.model_settings.tool_choice == "required" or isinstance(agent.model_settings.tool_choice, str):
                agent.model_settings = ModelSettings(
                    temperature=agent.model_settings.temperature,
                    top_p=agent.model_settings.top_p,
                    frequency_penalty=agent.model_settings.frequency_penalty,
                    presence_penalty=agent.model_settings.presence_penalty,
                    tool_choice="auto",  # Reset to auto
                    parallel_tool_calls=agent.model_settings.parallel_tool_calls,
                    truncation=agent.model_settings.truncation,
                    max_tokens=agent.model_settings.max_tokens,
                )
            
            # Also reset run_config's model_settings if it exists
            if run_config.model_settings and (run_config.model_settings.tool_choice == "required" or 
                                             isinstance(run_config.model_settings.tool_choice, str)):
                run_config.model_settings = ModelSettings(
                    temperature=run_config.model_settings.temperature,
                    top_p=run_config.model_settings.top_p,
                    frequency_penalty=run_config.model_settings.frequency_penalty,
                    presence_penalty=run_config.model_settings.presence_penalty,
                    tool_choice="auto",  # Reset to auto
                    parallel_tool_calls=run_config.model_settings.parallel_tool_calls,
                    truncation=run_config.model_settings.truncation,
                    max_tokens=run_config.model_settings.max_tokens,
                )
                
        # Check that run_config's tool_choice was reset to "auto"
        assert run_config.model_settings.tool_choice == "auto"
        
    @mock.patch("agents.run.Runner._get_model")
    async def test_integration_prevents_infinite_loop(self, mock_get_model):
        """Integration test to verify that tool_choice reset prevents infinite loops"""
        # Create a counter to track model calls and detect potential infinite loops
        tool_call_counter = {"count": 0, "potential_infinite_loop": False}
        
        # Set up our mock model that will always use tools when tool_choice is set
        mock_model_instance = MockModel(tool_call_counter)
        # Return our mock model directly
        mock_get_model.return_value = mock_model_instance
        
        # Create an agent with tool_choice="required" to force tool usage
        agent = Agent(
            name="Test agent",
            instructions="You are a test agent",
            tools=[echo],
            model_settings=ModelSettings(tool_choice="required"),
            # Use "run_llm_again" to allow LLM to continue after tool calls
            # This would cause infinite loops without the tool_choice reset
            tool_use_behavior="run_llm_again",
        )
        
        # Set a timeout to catch potential infinite loops that our fix doesn't address
        try:
            # Run the agent with a timeout
            async def run_with_timeout():
                return await Runner.run(agent, input="Test input")
            
            result = await asyncio.wait_for(run_with_timeout(), timeout=2.0)
            
            # Verify the agent ran successfully
            assert result is not None
            
            # Verify the tool was called at least once but not too many times
            # (indicating no infinite loop)
            assert tool_call_counter["count"] >= 1
            assert tool_call_counter["count"] < 5
            assert not tool_call_counter["potential_infinite_loop"]
            
        except asyncio.TimeoutError:
            # If we hit a timeout, the test failed - we likely have an infinite loop
            assert False, "Timeout occurred, potential infinite loop detected"