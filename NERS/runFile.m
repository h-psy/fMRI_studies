function runFile(script_filenames)
try
    % Construct the command to run MATLAB script
    cmd = sprintf('matlab -r "run(''%s'')"', script_filenames);
    % Execute the command using the system function
    status = system(cmd);
    % Check if an error occurred during script execution
    if status ~= 0
        error('Error occurred while running %s', script_filenames);
    end
catch err
    fprintf('%s\n', err.message);
end
exit;
end
