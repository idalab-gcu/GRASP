import os
os.environ["DGL_USE_GRAPHBOLT"] = "0"
import dgl

import argparse
import numpy as np
import pandas as pd
import torch
import pickle
import time
import torch.cuda.amp as amp # AMP

import myutils
from grasp import LinkPredict
from data_loader import Data

def main(args):
    # 1. Load Data Paths
    train_path = f'{args.data}/train.tsv'
    valid_path = f'{args.data}/valid.tsv'
    test_path = f'{args.data}/test.tsv'
    text_embedding_path = f'{args.data}/{args.text_embedding_file}'
    knowledge_embedding_path = f'{args.data}/{args.knowledge_embedding_file}'
    
    freeze = args.freeze

    # 2. Read Triples
    train = pd.read_csv(train_path, sep='\t', header=None)
    valid = pd.read_csv(valid_path, sep='\t', header=None)
    test = pd.read_csv(test_path, sep='\t', header=None)
    graph = pd.concat([train, valid, test])

    # 3. Load Embeddings
    print("Loading Pretrained Embeddings files...")
    try: text_embeddings = np.load(text_embedding_path)
    except: 
        print(f"Warning: Text embeddings not found at {text_embedding_path}")
        text_embeddings = None
        
    try: ontology_embeddings = np.load(knowledge_embedding_path)
    except: 
        print(f"Warning: Domain embeddings not found at {knowledge_embedding_path}")
        ontology_embeddings = None
        

    if (args.w_text is None) and (args.w_domain is None):
        w_text = args.w
        w_domain = 1.0 - args.w
    else:
        w_text = 0.5 if args.w_text is None else args.w_text
        w_domain = 0.5 if args.w_domain is None else args.w_domain
        
        s = w_text + w_domain
        if s <= 0: 
            w_text, w_domain = 0.5, 0.5
        else: 
            w_text, w_domain = w_text / s, w_domain / s
            
    # 5. Process Graph Data
    print("Data Processing...")
    knowledge_graph = Data(graph, train, valid, test)
    num_nodes, num_rels, num_edges = knowledge_graph.get_stats()
    
    train_data_np = knowledge_graph.train_data
    valid_data_np = knowledge_graph.valid_data
    test_data_np = knowledge_graph.test_data
    total_data_np = knowledge_graph.total_data
    total_data = torch.LongTensor(total_data_np) 

    # 6. Initialize Model (GAM-UAE LinkPredict)
    model = LinkPredict(
        input_dim=num_nodes, 
        hidden_dim=args.n_hidden, 
        num_relations=num_rels, 
        num_bases=args.num_bases,
        num_hidden_layers=args.num_hidden_layers, 
        dropout=args.dropout,
        use_cuda=args.use_cuda, 
        regularization_param=args.reg_param,
        pretrained_text_embeddings=text_embeddings,
        pretrained_domain_embeddings=ontology_embeddings,
        freeze=freeze,
        w_text=w_text, 
        w_domain=w_domain
    )
    
    if torch.cuda.is_available() and args.use_cuda: device = torch.device("cuda")
    else: device = torch.device('cpu')

    model = model.to(device)
    total_data = total_data.to(device)
    print(f"Device: {device}")

    # 7. Build Graphs
    # Training Graph
    train_graph, train_rel, train_norm = myutils.build_graph(num_nodes, num_rels, train_data_np)
    train_deg = train_graph.in_degrees(range(train_graph.number_of_nodes())).float().view(-1, 1)
    adj_list = myutils.get_adj(num_nodes, train_data_np)

    # Validation Graph
    valid_graph, valid_rel, valid_norm = myutils.build_graph(num_nodes, num_rels, valid_data_np)
    valid_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1).to(device)
    valid_rel = torch.from_numpy(valid_rel).to(device)
    valid_norm = myutils.node_norm_2_edge_norm(valid_graph, torch.from_numpy(valid_norm).view(-1, 1)).to(device)
    valid_graph = valid_graph.to(device)
    valid_data_tensor = torch.LongTensor(valid_data_np).to(device)

    # Test Graph
    test_graph, test_rel, test_norm = myutils.build_graph(num_nodes, num_rels, test_data_np)
    test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1).to(device)
    test_rel = torch.from_numpy(test_rel).to(device)
    test_norm = myutils.node_norm_2_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1)).to(device)
    test_graph = test_graph.to(device)
    test_data_tensor = torch.LongTensor(test_data_np).to(device)

    # 8. Optimizer & Scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = amp.GradScaler() 

    # 9. Training Loop
    print(f"Start training... (Batch: {args.graph_batch_size}, Neg: {args.negative_sample}, Patience: {args.patience})")
    best_mrr = 0.0
    best_epoch = 0
    patience_counter = 0 

    start_time = time.time()
    for iteration in range(1, 1 + args.iterations):
        model.train()

        # Sampling
        g, node_id, edge_type, node_norm, data, labels = \
            myutils.generate_sampled_graph_and_labels(
                train_data_np, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, train_deg, args.negative_sample,
                args.edge_sampler)

        node_id = torch.from_numpy(node_id).view(-1, 1).long().to(device)
        edge_type = torch.from_numpy(edge_type).to(device)
        edge_norm = myutils.node_norm_2_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1)).to(device)
        data = torch.from_numpy(data).to(device)
        labels = torch.from_numpy(labels).to(device)
        g = g.to(device)

        # Forward & Backward
        optimizer.zero_grad()
        with amp.autocast():
            embed = model(g, node_id, edge_type, edge_norm)
            loss = model.get_loss(g, embed, data, labels)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # === Validation & Early Stopping ===
        if iteration % args.evaluate_every == 0:
            eval_start = time.time()
            # print(f"\n-------------------------- EVAL {iteration} START --------------------------")
            
            model.eval()
            with torch.no_grad():
                output_val = model(valid_graph, valid_node_id, valid_rel, valid_norm)
                _, val_mrr, _ = myutils.calc_mrr(
                    output_val, model.relation_weights, valid_data_tensor,
                    total_data,
                    batch_size=args.eval_batch_size,
                    neg_sample_size_eval=args.neg_sample_size_eval,
                    hits=[1, 3, 10], eval_p=args.eval_protocol
                )
            print(f"Epoch {iteration} | Loss {loss.item():.5f} | MRR: {val_mrr:.6f} | Best: {best_mrr:.6f}")
            print(f"  > Validation MRR: {val_mrr:.6f} (Best: {best_mrr:.6f})")

            if val_mrr > best_mrr:
                best_mrr = val_mrr
                best_epoch = iteration
                patience_counter = 0 
                
                
                torch.save({'state_dict': model.state_dict(), 'iteration': iteration, 'mrr': best_mrr}, args.model_state_file)
                # print(f"  >>> New Best Model Saved!")
            else:
                patience_counter += 1 
                # print(f"  [Patience] No improvement for {patience_counter}/{args.patience} checks.")
                
                if patience_counter >= args.patience:
                    print(f"\n[Early Stopping] Triggered! No improvement for {args.patience} consecutive evaluations.")
                    break 

            model.train()
            # print(f"\nTraining {iteration} Time: {time.time()-start_time:.1f}s")
            # print(f"EVAL END Time: {time.time()-eval_start:.1f}s")

    end_time = time.time()
    total_train_time = end_time - start_time

    print(f"\n\nTraining Finished.")
    print(f"Loading Best Model (from Epoch {best_epoch} with MRR {best_mrr:.6f})...")

    # 10. Final Testing
    checkpoint = torch.load(args.model_state_file)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    with torch.no_grad():
        infer_start = time.time()
        output_test = model(test_graph, test_node_id, test_rel, test_norm)
        mr, mrr, hits_dict = myutils.calc_mrr(
            output_test, model.relation_weights, test_data_tensor,
            total_data,
            batch_size=args.eval_batch_size, 
            neg_sample_size_eval=args.neg_sample_size_eval,
            hits=[1, 3, 10], eval_p=args.eval_protocol
        )
        
        infer_end = time.time()

    infer_time = infer_end - infer_start
    num_test_triples = test_data_tensor.shape[0]
    throughput = (2 * num_test_triples) / infer_time if infer_time > 0 else 0.0

    print(f"Inference Time (s): {infer_time:.6f}")
    print(f"Inference Throughput (triples/s): {throughput:.4f}")
    
    print("\n[Final Test Results using Best Valid Model]")
    print(f"MR: {mr:.6f}")
    print(f"MRR: {mrr:.6f}")
    for key, value in hits_dict.items():
        print(f"Hits @ {key} = {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data", dest="data", default="hetionet", help="Data folder")
    parser.add_argument("--text_embedding_file", dest="text_embedding_file", default="pubmedbert_embeddings_768.npy")
    parser.add_argument("--knowledge_embedding_file", dest="knowledge_embedding_file", default="poincare_embeddings.npy")
    # [제거됨] global_embedding_file
    
    parser.add_argument("--freeze", action="store_true", help="Freeze pretrained embeddings")
    
    parser.add_argument("--w", dest="w", type=float, default=0.5, help="Weight for Text (1-w for Domain) if specific weights not given")
    parser.add_argument("--w_text", type=float, default=None)
    parser.add_argument("--w_domain", type=float, default=None)
    # [제거됨] w_global

    parser.add_argument("--n_hidden", dest="n_hidden", type=int, default=512) 
    parser.add_argument("--num_bases", dest="num_bases", type=int, default=32) 
    parser.add_argument("--num_hidden_layers", dest="num_hidden_layers", type=int, default=2)
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.2)
    parser.add_argument("--use_cuda", dest="use_cuda", type=bool, default=True)
    parser.add_argument("--reg_param", dest="reg_param", type=float, default=0.01)

    parser.add_argument("--iterations", dest="iterations", type=int, default=40000)
    parser.add_argument("--evaluate_every", dest="evaluate_every", type=int, default=30)
    parser.add_argument("--lr", dest="lr", type=float, default=0.002) 
    parser.add_argument("--graph_batch_size", dest="graph_batch_size", type=int, default=250)  # 5000
    parser.add_argument("--graph_split_size", dest="graph_split_size", type=float, default=0.5)
    parser.add_argument("--negative_sample", dest="negative_sample", type=int, default=100) 
    parser.add_argument("--edge_sampler", dest="edge_sampler", default="uniform")
    parser.add_argument("--grad_norm", dest="grad_norm", type=float, default=1.0)
    
    parser.add_argument("--patience", dest="patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--eval_batch_size", dest="eval_batch_size", type=int, default=10000)
    parser.add_argument("--neg_sample_size_eval", dest="neg_sample_size_eval", type=int, default=50)
    parser.add_argument("--eval_protocol", dest="eval_protocol", default="filtered")
    parser.add_argument("--model_state_file", dest="model_state_file", default="./hetionet/test_time_hetionet.pth")

    args = parser.parse_args()
    main(args)
