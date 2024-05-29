python main.py --d_model 512 --n_head 4 --n_ffn_layers 2 --n_encoder_layers 4 --n_decoder_layers 4 --dropout 0.2 --batch_size 16 --num_epochs 50 \
               --model_path ./Epoch:0_best_encoder_model.pth

# parser.add_argument('--model_type', type=str, default='encoder_only', help='model type: encoder_only, decoder_only, encoder_decoder')
# parser.add_argument('--model_path', type=str, default=None, help='path to load model')
# parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
# parser.add_argument('--n_head', type=int, default=4, help='number of heads in multi-head attention')
# parser.add_argument('--n_ffn_layers', type=int, default=2, help='number of layers in FFN')
# parser.add_argument('--n_encoder_layers', type=int, default=4, help='number of layers in encoder')
# parser.add_argument('--n_decoder_layers', type=int, default=4, help='number of layers in decoder')
# parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
# parser.add_argument('--max_len', type=int, default=512, help='maximum length of input sequence')
# parser.add_argument('--batch_size', type=int, default=2, help='batch size')
# parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
# parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')