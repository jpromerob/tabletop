
with aestream.UDPInput((res_y, res_x), device = 'cpu', port=args.port) as stream1:
            
    while True:

        reading = stream1.read()

        reshaped_data = torch.tensor(np.transpose(reading), dtype=torch.float32).unsqueeze(0)
        device_input_data = reshaped_data.to(device)

        out_t_dev, out_x_dev, out_y_dev = model(device_input_data)

        torch.cuda.synchronize() 

        out_x = out_x_dev.cpu().squeeze().numpy()
        out_y = out_y_dev.cpu().squeeze().numpy()

        