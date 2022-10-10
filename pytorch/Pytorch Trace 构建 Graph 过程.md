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

c++ 中 trace 函数：

```c++
std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    Module* self) {
  try {

    auto state = std::make_shared<TracingState>();
    // setTracingState 将state 这个实例set下来，在之后计算节点get出来insert计算过程
    setTracingState(state);

    // state这个数据结构会在forward过程中存储trace到的计算过程
    if (self) {
      Value* self_value = state->graph->insertInput(0, "self")->setType(
          self->_ivalue()->type());
      gatherParametersAndBuffers(state, self_value, *self, {"__module"});
    }

    for (IValue& input : inputs) {
      input = addInput(state, input, input.type(), state->graph->addInput());
    }
    auto graph = state->graph;
    //　将python中的变量名解析函数绑定下来
    getTracingState()->lookup_var_name_fn = std::move(var_name_lookup_fn);
    getTracingState()->strict = strict;
    getTracingState()->force_outplace = force_outplace;

    // 开始forward，在计算发生时，会把计算记录到state中
    auto out_stack = traced_fn(inputs);

    // Exit a trace, treating 'out_stack' as the outputs of the trace.  These
    // are the variables whose values will be computed upon subsequent
    // invocations of the trace.
    size_t i = 0;
    for (auto& output : out_stack) {
      // NB: The stack is in "reverse" order, so when we pass the diagnostic
      // number we need to flip it based on size.
      state->graph->registerOutput(
          state->getOutput(output, out_stack.size() - i));
      i++;
    }
    setTracingState(nullptr);

    if (getInlineEverythingMode()) {
      Inline(*graph);
    }
    FixupTraceScopeBlocks(graph, self);
    NormalizeOps(graph);
    return {state, out_stack};
  } catch (...) {
    tracer::abandon();
    throw;
  }
}
```

步骤大致如下：
- 创建一个 TracingState 对象来保存 trace 中记录的计算过程
- 逐步 trace 每个计算 ，即把 Node(OP) 添加到 Graph 中
- 完成 trace 过程

### 1.3 TracingState

```c++
struct TORCH_API TracingState
    : public std::enable_shared_from_this<TracingState> {
  TracingState();
  ~TracingState();

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::shared_ptr<Graph> graph;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool warn = true;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool strict = true;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  bool force_outplace = false;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::function<std::string(const Variable& var)> lookup_var_name_fn =
      [](const Variable& var) { return ""; };

  void enterFrame() {
    env_stack.emplace_back();
  }

  void leaveFrame() {
    env_stack.pop_back();
  }

  void setValue(const IValue& v, Value* value);
  void delValue(const IValue& var);
  Value* getValue(const IValue& var);
  Value* getOutput(const IValue& var, size_t i);
  bool hasValue(const IValue& var) const;

  Node* createNode(c10::Symbol op_name, size_t num_outputs);
  void insertNode(Node* node);

 private:
  using WeakIValue = at::WeakIValue;

  struct WeakIValueHasher {
    size_t operator()(const WeakIValue& t) const {
      return t.hash();
    }
  };

  struct WeakIValueEq {
    bool operator()(const WeakIValue& t1, const WeakIValue& t2) const {
      return t1.isSameIdentity(t2);
    }
  };

  using Frame =
      std::unordered_map<WeakIValue, Value*, WeakIValueHasher, WeakIValueEq>;
  std::vector<Frame> env_stack;
};
```

TracingState 用来保存 trace 过程中的数据结构，同时也会实现 ivalue 和 value 之间的转换。

## 2. 添加 Node(OP) 过程
```c++
torch::jit::Node* node = nullptr;
std::shared_ptr<jit::tracer::TracingState> tracer_state;
if (jit::tracer::isTracing()) {
        tracer_state = jit::tracer::getTracingState();
        at::Symbol op_name;
        op_name = jit::Symbol::fromQualString("aten::__ilshift__");
        node = tracer_state->graph->create(op_name, /*num_outputs=*/0);
        jit::tracer::recordSourceLocation(node);
        jit::tracer::addInputs(node, "self", self);
        jit::tracer::addInputs(node, "other", other);
        tracer_state->graph->insertNode(node);

        jit::tracer::setTracingState(nullptr);
}
TypeDefault::__ilshift__(self, other);
if (tracer_state) {
        jit::tracer::setTracingState(std::move(tracer_state));
        jit::tracer::addOutput(node, self);
}
```

大致分为以下几个步骤：
- 获取当前的 TracingState
- 新建当前 OP 的 Symbol
    > A Symbol is like an interned string, but with a little extra structure; it is namespaced via SymbolNamespace and the resulting intern pointers support efficient namespace testing.

    我理解 Symbol 类似是一个 OP 的标识，或者直观理解为就是一个字符串 ，常见的 Symbol 例如：`prim::add` 等，可以通过 Symbol 来创建对应的 Node 的名字。

- 调用 Graph 的 Create 方法，创建 OP 对应的 Node，该方法只是简单的新建一个 Node 对象，并根据传入的参数，给新建的 Node 对象添加 output，即新构建一个 value 对象。

    ```c++
    // Graph 的 create 方法
    Node* Graph::create(NodeKind kind, size_t num_outputs) {
    // NB: Node constructor adds node to all_nodes
    auto n = new Node(this, kind);
    for (const auto i : c10::irange(num_outputs)) {
        (void)i;
        n->addOutput();
    }
    return n;
    }

    // Node 的 addOutput 方法
    Value* Node::addOutput() {
    outputs_.push_back(new Value(this, outputs_.size()));
    op_ = nullptr;
    return outputs_.back();
    }

    // value 的构造函数，offset 应该是对应的第几个输入/输出
    inline Value::Value(Node* node_, size_t offset_)
        : node_(node_),
        offset_(offset_),
        unique_(node_->graph_->next_unique_++),
        type_(TensorType::get()) {
    node_->graph_->all_values.emplace(this);
    }
    ```

- 记录该 OP 对应的源代码位置，jit::tracer::recordSourceLocation(node);，原理待分析
- 给新创建的 Node 创建对应的 Input
  这一步实际最终会调用 Node 的 addInput 方法来实现，在 Tracer 封装的 addInput 方法里，还会实现 IValue 到 value 的转化
    ```c++
    // tracer 的 addInputs 方法
    void addInputs(Node* n, const char* name, const at::Tensor& value) {
    n->addInput(getValueTrace(value));
    }

    // node 的 addInput 方法
    Value* Node::addInput(Value* value) {
    AT_ASSERT(graph_ == value->owningGraph());
    op_ = nullptr;
    value->uses_.emplace_back(this, inputs_.size());
    inputs_.push_back(value);
    return value;
    }

    // Given a IValue 'var', return the 'node' which represents the instruction
    // which computes the value of this variable in the IR.
    // Here, we interpret untraced variables as constants that are just embedded
    // in the graph.  This is useful to handle code which does things like this
    // (from torch.autograd.variable, now moved to C++):
    //
    //    def mm(self, matrix):
    //      output = Variable(self.data.new(self.data.size(0), matrix.data.size(1)))
    //      return Addmm.apply(output, self, matrix, 0, 1, True)
    //
    // Here, mm fakes up a dummy variable with uninitialized data to do an inplace
    // update on, but subsequently ignores it because the alpha scaling factor is
    // zero. This is one of the cases where a Variable can be created inside of a
    // trace, and if we treat it as a constant, everything will work out.
    Value* getValueTrace(const IValue& var) {
    return getTracingState()->getValue(var);
    }
    ```

- 将创建好的 Node 插入到 Graph 中
    ```c++
    // insert before insert_before_ node
    // initialized to insert at the end of the top level block
    // can be changed with setInsertPoint()
    Node* insertNode(Node* n) {
        AT_ASSERT(
            insert_before_->inBlockList() &&
            "insert point node is no longer in a block list");
        return n->insertBefore(insert_before_);
    }
    ```
- 运行结束后，将输出添加到 Node 中
## 3. 参考文档
- [JIT Technical Overview](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md)
- [PyTorch 源码解读之即时编译篇](https://zhuanlan.zhihu.com/p/361101354)
- [TorchScript 如何实现Python -> C++ 代码转换](https://zhuanlan.zhihu.com/p/361101354)
- [TorchScript 入门篇](http://www.zh0ngtian.tech/posts/76ff5f2a.html)