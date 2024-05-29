import argparse

import torch
from encoder_only import SimplifiedBERT
from decoder_only import SimplifiedGPT
from encoder_decoder import SimpleEncoderDecoder
from load_dataset import get_train_dev_dataloader, load_test_dataset
from train_test import Trainer, evaluate_encoder_model, evaluate_decoder_model


def main(args) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load dataset
    train_dataloader, dev_dataloader, label2id, vocab = get_train_dev_dataloader(
        "./ner-data", args.model_type, args.max_len, args.batch_size)
    test_dataloader = load_test_dataset("./ner-data")

    input_dim = len(vocab)
    output_dim = len(label2id)
    
    # # load model
    if args.model_type == 'encoder_only':
        model = SimplifiedBERT(input_dim, args.d_model, args.n_head, args.n_ffn_layers, args.n_encoder_layers, args.max_len, output_dim, args.dropout)
    elif args.model_type == 'decoder_only':
        model = SimplifiedGPT(input_dim, args.d_model, args.n_head, args.n_ffn_layers, args.n_decoder_layers, args.max_len, output_dim, args.dropout)
    elif args.model_type == 'encoder_decoder':
        model = SimpleEncoderDecoder(input_dim, args.d_model, args.n_head, args.n_ffn_layers,
                                    args.n_encoder_layers, args.n_decoder_layers, args.max_len, output_dim, args.dropout)
    else:
        raise ValueError("Invalid model type")
    model.to(device)
    
    # load model from path
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    
    # train model
    trainer = Trainer(model, args.model_type, train_dataloader, dev_dataloader, test_dataloader, num_epochs=args.num_epochs, lr=args.lr, device=device)
    trainer.run(args.multi_gpu)
    # evaluate_decoder_model(model, dev_dataloader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='decoder_only', help='model type: encoder_only, decoder_only, encoder_decoder')
    parser.add_argument('--model_path', type=str, default=None, help='path to load model')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_head', type=int, default=4, help='number of heads in multi-head attention')
    parser.add_argument('--n_ffn_layers', type=int, default=2, help='number of layers in FFN')
    parser.add_argument('--n_encoder_layers', type=int, default=4, help='number of layers in encoder')
    parser.add_argument('--n_decoder_layers', type=int, default=4, help='number of layers in decoder')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--max_len', type=int, default=512, help='maximum length of input sequence')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--multi_gpu', type=bool, default=False, help='use multi-gpu training')
    # parser.add_argument('--device', type=str, default='cuda', help='device to use')
    # parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()
    
    main(args)
