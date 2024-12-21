import numpy as np
import matplotlib.pyplot as plt

# Function that defines the ODE dy/dx = f(x, y)
def f(x, y):
    return y  # In this case, f(x, y) = y

# Runge-Kutta 4th order method
def runge_kutta(x0, y0, h, steps):
    x_values = [x0]
    y_values = [y0]

    x = x0
    y = y0

    for n in range(steps):
        k1 = h * f(x, y)
        k2 = h * f(x + h / 2, y + k1 / 2)
        k3 = h * f(x + h / 2, y + k2 / 2)
        k4 = h * f(x + h, y + k3)

        # Update y and x
        y += (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h

        # Store the results
        x_values.append(x)
        y_values.append(y)

    return x_values, y_values

# Main function to take user input and plot the results
def main():
    # Take input from the user
    x0 = float(input("Enter initial value x0: "))
    y0 = float(input("Enter initial value y0: "))
    h = float(input("Enter step size h: "))
    steps = int(input("Enter number of steps: "))

    # Call the Runge-Kutta function
    x_values, y_values = runge_kutta(x0, y0, h, steps)

    # Plotting the results
    plt.plot(x_values, y_values, marker='o')
    plt.title('Runge-Kutta 4th Order Method')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
