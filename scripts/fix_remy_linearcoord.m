clear variables

for day = 35:37
    load(sprintf('remypos%d.mat', day))
    load(sprintf('remytask%d.mat', day))
    
    for epoch = [2, 4]

        data = pos{1, day}{1, epoch}.data;

        left_arm = [122.5, 48.97; 123.8, 128; 161.8, 129.1; 157.8, 48.97];
        left_arm = cat(3, left_arm, left_arm);
        linearcoord{1} = left_arm;

        right_arm = [122.5, 48.97; 123.8, 128; 87.7, 128.4; 87.02, 48.97];
        right_arm = cat(3, right_arm, right_arm);
        linearcoord{2} = right_arm;

        task_info = task{1, day}{1, epoch};
        task_info.('linearcoord') = linearcoord;
        task{1, day}{1, epoch} = task_info;
    end



    save(sprintf('remytask%d.mat', day), 'task', '-v7')
end