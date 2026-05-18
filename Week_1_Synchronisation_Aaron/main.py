import questionary

operation = questionary.select('Select operation:', choices=['Generate key', 'Record signal']).ask()

if operation == 'Generate key':
    from Generator_key_only import main as generate_key_main
    generate_key_main()
elif operation == 'Record signal':
    from Recieving_signal import main as record_signal_main
    record_signal_main()
elif operation == 'Compare signals':
    from Recieve_chirp import main as compare_signals_main
    compare_signals_main()