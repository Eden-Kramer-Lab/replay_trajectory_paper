clear variables;

for day = 35:37
    load(sprintf('remypos%d.mat', day))
    for epoch = 1:5
        pos{1, day}{1, epoch}.fields = 'time x y dir vel x-sm y-sm dir-sm vel-sm';
    end
    save(sprintf('remypos%d.mat', day), 'pos', '-v7')
end