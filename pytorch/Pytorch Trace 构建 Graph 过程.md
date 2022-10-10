# Pytorch Trace 构建 Graph 过程

## 1. 从 Python 到 C++

> The tracer produces graphs by recording what actual operations are done on `Tensors`.
The entry point from Python into C++ for tracing using `torch.jit.trace` is `_create_method_from_trace`.

> A thread local instance of the TracingState object maintains a mapping between actual data being computed during the trace (e.g. Tensors) stored in `IValues`, and the abstract `Value` in the `Graph` that would compute each value. The functions `void setValueTrace(const IValue&, Value*)` and `Value* getValueTrace(const IValue&)` are used by the tracer to maintain this mapping.

> An initial `IValue` to `Value` mapping is set up between the inputs to the function being traced and symbolic `Value` inputs to the `Graph` being constructed. If we are tracing a `torch.nn.Module`, the tracer also adds Parameters and sub-Modules to the Module being constructed that correspond to the Python `torch.nn.Module` being traced.  Mappings for these values are also added so that uses of the Parameters in the trace will create uses of the Parameters in the `Graph`.

> As the trace runs, individual operators create `Nodes` in the `Graph` being traced to record what happens. This code is currently generated per operator in tools/autograd/gen_variable_type.py.

### 1.1 trace 模式使用

```C++ {.line-numbers}
class Module_0(torch.nn.Module):
    def __init__(self, N, M):
        super(Module_0, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))
        self.linear = torch.nn.Linear(N, M)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.weight.mm(input)
        output = self.linear(output)
        return output


scripted_module = torch.jit.trace(Module_0(2, 3).eval(), (torch.zeros(3, 2)))
scripted_module.save("Module_0.pt")
```

对于只有 Tensor 操作的模型，比较适合使用 trace 模式。

### 1.2 trace 实现

```c++
func,
    example_inputs,
    optimize=None,
    check_trace=True,
    check_inputs=None,
    check_tolerance=1e-5,
    strict=True,
    _force_outplace=False,
    _module_class=None,
    _compilation_unit=_python_cu,
):


    # 发现是nn.Module instacene forward, 追踪forward
    if isinstance(func, torch.nn.Module):
        return trace_module(
            func,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
        )
    # 传进来的是某个module instance的forward
    if (
        hasattr(func, "__self__")
        and isinstance(func.__self__, torch.nn.Module)
        and func.__name__ == "forward"
    ):
        return trace_module(
            func.__self__,
            {"forward": example_inputs},
            None,
            check_trace,
            wrap_check_inputs(check_inputs),
            check_tolerance,
            strict,
            _force_outplace,
            _module_class,
        )
    # 一个查找变量名的接口
    var_lookup_fn = _create_interpreter_name_lookup_fn(0)

    # C++ 入口 
    traced = torch._C._create_function_from_trace(
        name, func, example_inputs, var_lookup_fn, strict, _force_outplace
    )

    # 检查traced 与 原func是否有差异
    if check_trace:
        if check_inputs is not None:
            _check_trace(
                check_inputs,
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
            )
        else:
            _check_trace(
                [example_inputs],
                func,
                traced,
                check_tolerance,
                strict,
                _force_outplace,
                False,
                _module_class,
            )

    return traced
```

c++的入口就是 _create_function_from_trace

```c++
traced = torch._C._create_function_from_trace(
        name, func, example_inputs, var_lookup_fn, strict, _force_outplace
      )
```