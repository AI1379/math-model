clc
clear

function plot_ode(t, y, title_str)
    figure
    plot(t, y, 'LineWidth', 2)
    title(title_str)
    xlabel('Time')
    ylabel('Solution')
    grid on
end

% Non-stiff ODE example
[t, y] = ode45(@func1, [0, 0.5], 1)
plot_ode(t, y, 'Non-stiff ODE Example: dy/dx = -2y + 2x^2 + 2x')

% Stiff ODE example
[t, y] = ode45(@func2, [0, 1], [1; 0; -1])

plot(t, y(:, 1), '-', t, y(:, 2), '--', t, y(:, 3), ':', 'LineWidth', 2)
legend('y1', 'y2', 'y3')
