function dy = func2(~, y)
    dy = [y(2); y(3); 3 * y(3) + y(2) * y(1)];
end
