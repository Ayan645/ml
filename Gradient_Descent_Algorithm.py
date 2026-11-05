# Gradient Descent Algorithm
# Function: y = (x + 3)^2

def function(x):
    return (x + 3) ** 2


def derivative(x):
    return 2 * (x + 3)


# Starting point
x = 2

# Learning rate
learning_rate = 0.1

# Number of iterations
iterations = 100

print("Gradient Descent")
print("----------------")

for i in range(iterations):
    gradient = derivative(x)

    # Gradient Descent formula
    x = x - learning_rate * gradient

    print(f"Iteration {i + 1}: x = {x:.6f}, y = {function(x):.6f}")

print("\nLocal Minimum:")
print(f"x = {x:.6f}")
print(f"y = {function(x):.6f}")

# -----------------------------------------------------------------------------
# EXPLANATION (added):
# 1. The objective function is f(x) = (x + 3)^2. Its minimum is at x = -3,
#    where f(x) = 0.
# 2. function(x) returns the current objective-function value.
# 3. derivative(x) returns the slope, 2(x + 3), at the current x.
# 4. Each iteration moves x in the opposite direction of the slope:
#       new_x = old_x - learning_rate * derivative(old_x)
# 5. The learning rate controls the step size; iterations controls how many
#    updates are performed. The final printed x should be close to -3.
#
# RUN ON LINUX:
#   cd sem7_annotated/ml
#   python3 Gradient_Descent_Algorithm.py
# No external libraries or dataset are required.
# -----------------------------------------------------------------------------
