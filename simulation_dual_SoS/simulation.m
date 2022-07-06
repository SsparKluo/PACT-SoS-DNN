clear;

% addpath("/usr/local/Polyspace/R2020b/toolbox/k-Wave");

% Define grid
dx = 5e-5; % grid point spacing in the x direction [m]
dy = 5e-5; % grid point spacing in the y direction [m]
Nx = 512; %
Ny = 384; %
kgrid = kWaveGrid(Nx, dx, Ny, dy);
kgrid.setTime(901, 25e-9)

% Simulation loop
for i = 701:701
    clear("source");
    clear("medium");
    clear("sensor");

    % create filepath
    filepath = string(i) + "/";

    if ~exist(filepath, 'dir')
        mkdir(filepath)
    end

    % Define 'water-tissue boundary':
    % use bezier curve to draw the boundary
    points = randi([3, 5]);
    bezier_input_x = zeros(1, points);
    bezier_input_y = zeros(1, points);
    bezier_input_x(points) = Ny;
    bezier_input_y(1) = randi([96, 320]);
    bezier_input_y(points) = randi([96, 320]);
    last_x = 0;

    for temp = 2:(points - 1)
        bezier_input_x(temp) = randi([last_x, last_x + idivide(int32(Ny - last_x), ...
                                    int32(points - temp), 'floor')]);
        last_x = bezier_input_x(temp);
        bezier_input_y(temp) = randi([30, 320]);
    end

    [~, Y] = bezier(bezier_input_x, bezier_input_y);

    % Define medium properties
    medium.alpha_power = 0.75;
    medium.alpha_coeff = 1.5;

    medium_SoS = waterSoundSpeed(30); % acquire the SoS of water with T = 30 Celsius
    medium.sound_speed = medium_SoS * ones(Nx, Ny);
    medium.density = 1000 * ones(Nx, Ny);
    tissue_SoS = 1600 + 40 * randn;

    for y = 1:Ny
        medium.sound_speed(round(Y(y)):Nx, y) = tissue_SoS;
        medium.density(round(Y(y)):Nx, y) = 1058;
        % Agrawal, Sumit, et al. "Modeling combined ultrasound and photoacoustic imaging:
        % simulations aiding device development and artificial intelligence."
        % Photoacoustics 24 (2021): 100304.
    end

    % pressure source
    source.p0 = zeros(Nx, Ny);
    repeat_times = randi([20, 40]);

    % Skin or tissue reaction (No for now)

    for a = 1:repeat_times
        %locate the source
        y = randi([30, 354]);
        x = randi([round(Y(y)) + 40, 482]);

        % make a pressure source
        % width
        t = exp(- [51 2] / 2.5);
        width =- 6 * log(diff(t) * rand(1) + t(1));
        %elliptic ratio
        elliptic_ratio = 0.6 + rand(1) * 0.4;
        % pressure amplitude
        magnitude = 0.6 + rand(1) * 0.4;
        % angle
        theta = rand(1) * pi;

        % make a cover
        [columnsInImage, rowsInImage] = meshgrid(1:Ny, 1:Nx);
        % Next create the ellipse in the image.
        radiusY = width * sqrt(elliptic_ratio.^2) ./ 2;
        y_e = columnsInImage - y;
        x_e = rowsInImage - x;

        ellipsePixels = (x_e .* cos(theta) - y_e .* sin(theta)).^2 ./ ...
            ((width ./ 2).^2) + (x_e .* sin(theta) + y_e .* cos(theta)).^2 ./ ...
            (radiusY.^2) <= 1;
        % ellipsePixels is a 2D "logical" array.

        source.p0 = source.p0 + ellipsePixels .* magnitude;
    end

    % sensor_data simulation and add noise
    %sensor.mask = zeros(Nx, Ny);
    %sensor.mask(1, 1:6:768 ) = 1;

    sensor.mask = ones(4, 128);

    for a = 0:127
        sensor.mask(2, a + 1) = 3 * a + 1;
        sensor.mask(4, a + 1) = 3 * (a + 1);
    end

    sensor_data_mat = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);

    %sensor.mask = zeros(Nx, Ny);
    %sensor.mask(1, 1:6:768 ) = 1;
    %sensor_data = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);

    % process sensor_data
    sensor_data = zeros(128, 901);

    for a = 1:128

        for b = 1:901
            sensor_data(a, b) = sum(sensor_data_mat(a).p(1, :, 1, b));
            %sensor_data_mat(a).p(1,1:6,1,b) = sensor_data(a,b)./6;
        end

    end

    % add noise
    snr = randi([2 5]);
    noisy_sensor_data = addNoise(sensor_data, snr);
    save(filepath + "raw_data_128", 'noisy_sensor_data')

    %prepare for time reversal
    clearvars sensor;
    sensor.mask = zeros(Nx, Ny);
    sensor.mask(1, 1:384) = 1;
    reverse_sensor_data = zeros(384, 901);

    GT = medium.sound_speed;
    imwrite(source.p0, filepath + "s0.png");
    save(filepath + "GT", "GT");

    % Input
    for a = 1:384
        reverse_sensor_data(a, :) = sensor_data(idivide(int32(a), int32(3), 'ceil'), :);
    end

    medium.sound_speed = medium_SoS;
    source.p0 = 0;
    source.p = 0;
    % use the sensor points as sources in time reversal
    source.p_mask = sensor.mask;
    % time reverse and assign the data
    source.p = fliplr(reverse_sensor_data);
    % enforce, rather than add, the time-reversed pressure values
    source.p_mode = 'dirichlet';
    % set the simulation to record the final image (at t = 0)
    sensor.record = {'p_final'};
    % run the time reversal reconstruction
    p0_estimate = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);
    % apply a positivity condition
    p0_estimate.p_final = p0_estimate.p_final .* (p0_estimate.p_final > 0);

    for a = 1:3
        source = rmfield(source, 'p');
        % set the initial pressure to be the latest estimate of p0
        source.p0 = p0_estimate.p_final;
        % set the simulation to record the time series
        sensor = rmfield(sensor, 'record');
        % calculate the time series using the latest estimate of p0
        sensor_data2 = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);
        % calculate the error in the estimated time series
        data_difference = reverse_sensor_data - sensor_data2;

        % assign the data_difference as a time-reversal source
        source.p_mask = sensor.mask;
        source.p = fliplr(data_difference);
        source = rmfield(source, 'p0');
        source.p_mode = 'dirichlet';
        % set the simulation to record the final image (at t = 0)
        sensor.record = {'p_final'};
        % run the time reversal reconstruction
        p0_update = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);
        % add the update to the latest image
        p0_estimate.p_final = p0_estimate.p_final + p0_update.p_final;
        % apply a positivity condition
        p0_estimate.p_final = p0_estimate.p_final .* (p0_estimate.p_final > 0);
    end

    imwrite(p0_estimate.p_final, filepath + "input_128.png");

    clear("reverse_sensor_data");

    for a = 1:64

        for b = 1:3
            reverse_sensor_data((a - 1) * 3 + b, :) = noisy_sensor_data(2 * a - 1, :);
        end

    end

    save(filepath + "raw_data_64", 'reverse_sensor_data')

    save()

    for a = 1:64
        sensor.mask(1, 6 * (a - 1) + 4:6 * (a - 1) + 6) = 0;
    end

    source.p0 = 0;
    source.p = 0;
    medium.sound_speed = medium_SoS;
    clearvars "p0_estimate";
    % use the sensor points as sources in time reversal
    source.p_mask = sensor.mask;
    % time reverse and assign the data
    source.p = fliplr(reverse_sensor_data);
    % enforce, rather than add, the time-reversed pressure values
    source.p_mode = 'dirichlet';
    % set the simulation to record the final image (at t = 0)
    sensor.record = {'p_final'};
    % run the time reversal reconstruction
    p0_estimate = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);
    % apply a positivity condition
    p0_estimate.p_final = p0_estimate.p_final .* (p0_estimate.p_final > 0);

    for b = 1:3
        source = rmfield(source, 'p');
        % set the initial pressure to be the latest estimate of p0
        source.p0 = p0_estimate.p_final;
        % set the simulation to record the time series
        sensor = rmfield(sensor, 'record');
        % calculate the time series using the latest estimate of p0
        sensor_data2 = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);
        % calculate the error in the estimated time series
        data_difference = reverse_sensor_data - sensor_data2;

        % assign the data_difference as a time-reversal source
        source.p_mask = sensor.mask;
        source.p = fliplr(data_difference);
        source = rmfield(source, 'p0');
        source.p_mode = 'dirichlet';
        % set the simulation to record the final image (at t = 0)
        sensor.record = {'p_final'};
        % run the time reversal reconstruction
        p0_update = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);
        % add the update to the latest image
        p0_estimate.p_final = p0_estimate.p_final + p0_update.p_final;
        % apply a positivity condition
        p0_estimate.p_final = p0_estimate.p_final .* (p0_estimate.p_final > 0);
    end

    imwrite(p0_estimate.p_final, filepath + "input_64.png");

end
