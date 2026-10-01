"""One definition for tool dispatch, schemas, permissions, and scheduling metadata."""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Literal

from jsonschema.validators import validator_for


MODES = ('read-only', 'workspace-edit', 'execute')
MCP_READ_TOOLS = frozenset({'read_text_file', 'read_multiple_files', 'directory_tree', 'get_file_info',
                          'list_allowed_directories', 'search_files', 'list_directory_with_sizes'})


@dataclass(frozen=True)
class ToolSpec:
    schema: dict
    handler: Callable
    minimum_mode: str = 'read-only'
    side_effects: Literal['none', 'filesystem', 'process', 'network', 'unknown'] = 'unknown'
    concurrency: Literal['parallel', 'serial'] = 'serial'
    cacheable: bool = False
    compact_observation: bool = False
    task_kinds: tuple[str, ...] = ('all',)
    default_timeout: int | None = None
    max_timeout: int | None = None
    native: bool = True
    wait_for_process: bool = False
    validator: object = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.minimum_mode not in MODES:
            raise ValueError('Unknown tool permission mode')
        if self.side_effects not in ('none', 'filesystem', 'process', 'network', 'unknown'):
            raise ValueError('Unknown side-effect policy')
        if self.concurrency not in ('parallel', 'serial'):
            raise ValueError('Unknown concurrency policy')
        if self.cacheable and self.side_effects != 'none':
            raise ValueError('Only tools without side effects can reuse observations')
        if self.compact_observation and not self.cacheable:
            raise ValueError('Observation compaction requires a cacheable tool')
        if self.wait_for_process and self.side_effects != 'process':
            raise ValueError('Process waiting requires a process tool')
        definition = deepcopy(self.schema)
        parameters = definition['function'].get('parameters', {'type': 'object'})
        validator_class = validator_for(parameters)
        validator_class.check_schema(parameters)
        object.__setattr__(self, 'schema', definition)
        object.__setattr__(self, 'validator', validator_class(parameters))
        if self.default_timeout is not None:
            if self.max_timeout is None or not 1 <= self.default_timeout <= self.max_timeout:
                raise ValueError('Invalid timeout policy')
            declared = parameters.get('properties', {}).get('timeout_seconds', {})
            if declared.get('maximum') != self.max_timeout:
                raise ValueError('Timeout policy must match the tool schema')

    @property
    def name(self):
        return self.schema['function']['name']

    def permitted(self, mode):
        return MODES.index(mode) >= MODES.index(self.minimum_mode)


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, spec):
        if spec.name in self._tools:
            raise ValueError(f'Duplicate tool: {spec.name}')
        self._tools[spec.name] = spec

    def get(self, name):
        return self._tools.get(name)

    def require(self, name, mode):
        spec = self.get(name)
        if spec is None or not spec.permitted(mode):
            raise PermissionError(f'Tool {name} is unavailable in {mode} mode')
        return spec

    def validate(self, name, arguments):
        self._tools[name].validator.validate(arguments)

    def schemas(self, mode, task_kind='all'):
        return [deepcopy(spec.schema) for spec in self._tools.values()
                if spec.permitted(mode) and task_kind in spec.task_kinds]

    def execute(self, name, arguments, mode):
        spec = self.require(name, mode)
        self.validate(name, arguments)
        return spec.handler(arguments)
