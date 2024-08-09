import onnx
import onnx.helper
import onnx.numpy_helper
from onnx import TensorProto

# Path to the ONNX model
model_path = '/scratch1/rsawahn/results/all_best/export/ssd_sds_sliced.onnx'
modified_model_path = 'modified_model.onnx'

# Load the ONNX model
model = onnx.load(model_path)

def create_polynomial_approximation_nodes(node):
    x = node.input[0]
    y = node.output[0]

    # Polynomial coefficients for exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24
    coeffs = [1.0, 1.0, 0.5, 1/6, 1/24]

    # Create constant nodes for coefficients
    nodes = []
    initializer_names = []
    for i, coeff in enumerate(coeffs):
        initializer_name = f'{x}_coeff_{i}'
        initializer_tensor = onnx.helper.make_tensor(
            name=initializer_name,
            data_type=TensorProto.FLOAT,
            dims=[],
            vals=[coeff],
        )
        model.graph.initializer.append(initializer_tensor)
        initializer_names.append(initializer_name)

    # Create polynomial terms
    terms = [x]
    for i in range(1, len(coeffs)):
        term_mul = onnx.helper.make_node(
            'Mul',
            inputs=[terms[-1], x],
            outputs=[f'{x}_term_{i}']
        )
        terms.append(term_mul.output[0])
        nodes.append(term_mul)
        
        term_mul_coeff = onnx.helper.make_node(
            'Mul',
            inputs=[terms[-1], initializer_names[i]],
            outputs=[f'{x}_term_{i}_mul_coeff']
        )
        nodes.append(term_mul_coeff)

    # Create addition nodes to sum up the terms
    current_sum = initializer_names[0]
    for i in range(1, len(coeffs)):
        add_node = onnx.helper.make_node(
            'Add',
            inputs=[current_sum, f'{x}_term_{i}_mul_coeff'],
            outputs=[f'{x}_sum_{i}']
        )
        current_sum = f'{x}_sum_{i}'
        nodes.append(add_node)

    final_add_node = onnx.helper.make_node(
        'Add',
        inputs=[current_sum, x],
        outputs=[y]
    )
    nodes.append(final_add_node)

    return nodes

# Traverse the graph and replace Exp nodes
def replace_exp_with_polynomial_approximation(model):
    graph = model.graph
    new_nodes = []

    for node in graph.node:
        if node.op_type == 'Exp':
            print("replacing")
            print(node)
            approximation_nodes = create_polynomial_approximation_nodes(node)
            new_nodes.extend(approximation_nodes)
        else:
            new_nodes.append(node)

    # Clear the existing nodes and add the new nodes
    del graph.node[:]
    graph.node.extend(new_nodes)
    
        
# Replace Exp with polynomial approximation
replace_exp_with_polynomial_approximation(model)

# Save the modified model
onnx.save(model, modified_model_path)

print(f"Model saved to {modified_model_path}")
