clear;

% addpath("/usr/local/Polyspace/R2020b/toolbox/k-Wave");

% Define grid
dx = 5e-5; % grid point spacing in the x direction [m]
dy = 5e-5; % grid point spacing in the y direction [m]
Nx = 512; %
Ny = 384; %
kgrid = kWaveGrid(Nx, dx, Ny, dy);
kgrid.setTime(901, 25e-9)

for i = 15:701
    clear("source");
    clear("medium");
    clear("sensor");

    folder = './' + string(i) + '/';
    s0 = imread(folder + 's0.png');
    source.p0 = rescale(s0);
    

    % Define medium properties
    medium.alpha_power = 0.75;
    medium.alpha_coeff = 1.5;

    medium_SoS = waterSoundSpeed(30); % acquire the SoS of water with T = 30 Celsius
    medium.sound_speed = medium_SoS * ones(Nx, Ny);
    medium.density = 1000 * ones(Nx, Ny);

    sensor.mask = ones(4, 128);

    for a = 0:127
        sensor.mask(2, a + 1) = 3 * a + 1;
        sensor.mask(4, a + 1) = 3 * (a + 1);
    end

    sensor_data_mat = kspaceFirstOrder2DG(kgrid, medium, source, sensor, 'PMLInside', false);

    % process sensor_data
    sensor_data = zeros(128, 901);

    for a = 1:128

        for b = 1:901
            sensor_data(a, b) = sum(sensor_data_mat(a).p(1, :, 1, b));
            %sensor_data_mat(a).p(1,1:6,1,b) = sensor_data(a,b)./6;
        end

    end

    save(folder + "GT_Raw_128", 'sensor_data');
end