# training che usa kan -----------------------------------------------------------------------------------
def train_model(task_name, X_train, y_train, X_val, y_val,base_path,logs_path):
    # input_dim = X_train.shape[1] if task_name.startswith('overloaded') else X_train.shape[2]
    # input_dim = X_train.shape[2]
    # output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1  # Handle multi-output
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Split regression and classification targets
    y_val = np.array(y_val)
    y_train_reg, y_train_cls = y_train[:, :-1], y_train[:, -1]
    y_val_reg, y_val_cls = y_val[:, :-1], y_val[:, -1]

    input_dim = X_train.shape[2]
    reg_output_dim = y_train_reg.shape[1]
    num_permutations = 6

    model = build_janossy_rnn_model(reg_output_dim = reg_output_dim, input_dim=input_dim,
                                    rnn_type='gru', rnn_units=80, num_permutations=num_permutations )

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # Create the logs directory if it doesn't exist
    os.makedirs(logs_path, exist_ok=True)
    csv_logger = CSVLogger(logs_path + f'/{task_name}_logs.csv', append=False)


    # Training time tracking
    start_time = time.time()

    # Prepare the generator
    # train_sequence = JanossyPermutedBatchSequence( X_train, y_train, batch_size=32, num_permutations=num_permutations,shuffle_batch=True)
    X_train = prepare_janossy_input(X_train, num_permutations=num_permutations)
    X_val = prepare_janossy_input(X_val, num_permutations=num_permutations)

    history = model.fit(
        X_train,
        {'regression_output': y_train_reg, 'classification_output': y_train_cls},
        validation_data=(X_val, {'regression_output': y_val_reg, 'classification_output': y_val_cls}),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, lr_scheduler, csv_logger],
        verbose=1
    )

    training_time = time.time() - start_time
    # target_model_name = task_name + "_" + model_type + base_path.split("/")[-1]
    target_model_name = f"{task_name}_{os.path.basename(base_path)}"
    time_df.loc[target_model_name, MODEL_NAME] = training_time
    print(f"Training time for {target_model_name}: {training_time:.2f} seconds")

    # Save the model
    # path = f'{base_path}/{task_name}/{model_type}'
    os.makedirs(base_path, exist_ok=True)
    model.save(os.path.join(base_path, f"{task_name}.keras"))

    return model, history, training_time




def training(x_train_dict, x_val_dict, y_train_dict, y_val_dict, tasks, base_path, logs_path):
  trained_models = {}
  models_history = {}
  train_time = {}

  for task_name in tqdm(tasks):
    X_train = x_train_dict[task_name]
    X_val = x_val_dict[task_name]
    y_train = y_train_dict[task_name]
    y_val = y_val_dict[task_name]
    model, history, training_time = train_model(task_name, X_train, y_train, X_val, y_val,base_path, logs_path)
    trained_models[task_name] = model
    models_history[task_name] = history
    train_time[task_name] = training_time
  time_df.to_csv(training_time_path)
  return trained_models, models_history, train_time