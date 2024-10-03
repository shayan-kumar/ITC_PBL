function convolutional_code_example()
    % Define generator polynomials
    g1 = [1 0 1]; % g1(x) = x^2 + 1
    g2 = [1 1 1]; % g2(x) = x^2 + x + 1

    % Input data (binary)
    data = [1 0 1 1 0]; % Example input data
    disp('Input Data:');
    disp(data);

    % Encode the input data
    encoded_data = convolutional_encode(data, g1, g2);
    disp('Encoded Data:');
    disp(encoded_data);

    % Simulate received data (adding some noise/distortions if desired)
    % For now, just assume received data is exactly the encoded data (no noise)
    received_data = encoded_data; % Add noise/distortion if needed

    % Decode the received data using the Viterbi Algorithm
    decoded_data = viterbi_decode(received_data, g1, g2, length(data));
    disp('Decoded Data:');
    disp(decoded_data);
end

function encoded = convolutional_encode(data, g1, g2)
    % Initialize encoded data
    encoded = [];
    
    % Zero-pad the input data for convolution
    data = [data, zeros(1, length(g1) - 1)]; 

    % Perform convolution with the generator polynomials
    for i = 1:length(data) - (length(g1) - 1)
        segment = data(i:i + length(g1) - 1);
        % Apply generator polynomials and store the encoded result
        encoded = [encoded, mod(sum(segment .* g1), 2), mod(sum(segment .* g2), 2)];
    end
end

function decoded = viterbi_decode(received, g1, g2, data_length)
    % Number of states
    num_states = 2^(length(g1) - 1); % 4 states for a (2,1,3) convolutional code
    trellis = zeros(num_states, data_length + 1);
    path_metrics = inf(num_states, data_length + 1);
    path_metrics(1, 1) = 0; % Start at state 0

    % State transitions and their respective output bits
    state_transitions = [0 0; 1 0; 0 1; 1 1];

    % Viterbi algorithm
    for t = 1:data_length
        for s = 1:num_states
            for input = 0:1
                % Calculate the next state
                next_state = bitshift(s - 1, 1) + input;
                next_state = mod(next_state, num_states) + 1; % Ensure it wraps within num_states
                
                % Get binary representation of the state for convolution
                current_state_bits = bitget(s - 1, [2, 1]); % Get bits of current state
                
                % Calculate expected output based on the state and generator polynomials
                output1 = mod(sum(current_state_bits .* g1(2:end)) + input * g1(1), 2);
                output2 = mod(sum(current_state_bits .* g2(2:end)) + input * g2(1), 2);
                output = [output1, output2];

                % Calculate Hamming distance between received and expected output
                hamming_dist = sum(output ~= received(2*t-1:2*t));

                % Update path metrics and trellis if a lower metric is found
                metric = path_metrics(s, t) + hamming_dist;
                if metric < path_metrics(next_state, t + 1)
                    path_metrics(next_state, t + 1) = metric;
                    trellis(next_state, t + 1) = s;
                end
            end
        end
    end

    % Traceback to find the most likely path (decoding)
    [~, state] = min(path_metrics(:, end));
    decoded = zeros(1, data_length);
    
    for t = data_length:-1:1
        decoded(t) = bitshift(state - 1, -1); % Extract the most significant bit
        state = trellis(state, t + 1); % Backtrack through trellis
    end
end