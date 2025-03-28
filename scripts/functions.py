import ollama
import torch
import sqlite3
import io


def create_table(conn):
    c = conn.cursor()
    c.execute(
        '''
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            title TEXT,
            description TEXT,
            test_steps TEXT,
            model TEXT,
            embedding BLOB
        )
        '''
    )
    conn.commit()


def compute_embedding(entry):
    id, title, description, test_steps, model = entry
    input_text = [f"Name: {title}\n\nDescription: {description}\n\nTest Steps: {test_steps}"]
    responses = ollama.embed(model=model, input=input_text)
    embeddings = responses["embeddings"]
    embed_tensor = torch.tensor(embeddings).squeeze()
    return embed_tensor


def serialize_tensor(tensor):
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(blob):
    buffer = io.BytesIO(blob)
    return torch.load(buffer)


def get_entry_by_id(conn, entry_id):
    c = conn.cursor()
    c.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
    return c.fetchone()


def is_entry_different(existing_entry, new_entry):
    id, title, description, test_steps, model, blob = existing_entry
    n_id, n_title, n_description, n_test_steps, n_model = new_entry
    return (title != n_title or
            description != n_description or
            test_steps != n_test_steps or
            model != n_model)


def add_entry(conn, entry):
    id, title, description, test_steps, model = entry
    embedding = compute_embedding(entry)
    embedding_blob = serialize_tensor(embedding)
    c = conn.cursor()
    c.execute(
        "INSERT INTO entries (id, title, description, test_steps, model, embedding) VALUES (?, ?, ?, ?, ?, ?)",
        (id, title, description, test_steps, model, embedding_blob)
    )
    conn.commit()


def update_entry(conn, entry):
    id, title, description, test_steps, model = entry
    embedding = compute_embedding(entry)
    embedding_blob = serialize_tensor(embedding)
    c = conn.cursor()
    c.execute(
        "UPDATE entries SET title = ?, description = ?, test_steps = ?, model = ?, embedding = ? WHERE id = ?",
        (title, description, test_steps, model, embedding_blob, id)
    )
    conn.commit()


def get_embeddings_by_ids(conn, id_list):
    c = conn.cursor()
    placeholders = ', '.join('?' for _ in id_list)
    query = f"SELECT id, embedding FROM entries WHERE id IN ({placeholders})"
    c.execute(query, id_list)

    result_map = {}
    for row in c.fetchall():
        entry_id, blob = row
        embedding = deserialize_tensor(blob)
        result_map[entry_id]=embedding
    results = []
    for id in id_list:
        results.append((id, result_map[id]))
    return results

    # results = []
    # for row in c.fetchall():
    #     entry_id, blob = row
    #     embedding = deserialize_tensor(blob)
    #     results.append((entry_id, embedding))
    # return results
    #

def get_filtered_similarity_matrix(embed_tensor, block_size, similarity_index):
    # print(f"PURR 63866 emb {embed_tensor[63866]}")
    # print(f"PURR 64141 emb {embed_tensor[64141]}")

    num_embeds = embed_tensor.size(0)
    embed_size = embed_tensor.size(1)

    num_blocks = num_embeds // block_size + (1 if num_embeds % block_size else 0)

    filtered_list = []

    for i in range(num_blocks):
        i_start = i * block_size
        i_end = min((i + 1) * block_size, num_embeds)

        for j in range(num_blocks):
            j_start = j * block_size
            j_end = min((j + 1) * block_size, num_embeds)
            i_block = embed_tensor[i_start:i_end, :].to("cuda")
            j_block = embed_tensor[j_start:j_end, :].to("cuda")
            matrix = torch.matmul(i_block, j_block.T)

            # magic_i = 63866
            # magic_j = 64141

            # if i_start<=magic_i<i_end and j_start<=magic_j<j_end:
            #     print(f"PURR {matrix[magic_j-i_start][magic_i-j_start]}")

            # print(f"base matrix")
            # print(matrix)
            matrix[matrix < similarity_index] = 0
            # print(f"zerod matrix")
            # print(matrix)

            # Create a mask that only keeps entries where the global row index < global column index.
            global_rows = torch.arange(i_start, i_end, device="cuda").unsqueeze(1)  # shape (block_rows, 1)
            global_cols = torch.arange(j_start, j_end, device="cuda").unsqueeze(0)  # shape (1, block_cols)
            upper_mask = global_rows < global_cols
            # print(upper_mask)
            # Combine with the nonzero (above-threshold) condition.
            valid_mask = (matrix != 0) & upper_mask
            # print(f"valid mask")
            # print(valid_mask)

            # Retrieve indices and values where valid_mask is True.
            nonzero_indices = torch.nonzero(valid_mask, as_tuple=False)
            # print(nonzero_indices)
            scores = matrix[valid_mask]

            for k in range(nonzero_indices.shape[0]):
                r, c = nonzero_indices[k]
                global_row = i_start + r.item()
                global_col = j_start + c.item()
                score = scores[k].item()
                filtered_list.append((score, global_row, global_col))

    return filtered_list


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        # Initialize parent pointer for unseen elements
        if x not in self.parent:
            self.parent[x] = x
        # Path compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            # You can also use union by rank if needed for extra efficiency.
            self.parent[rootY] = rootX


def transitive_closure(edges):
    uf = UnionFind()
    # Process each edge to union the nodes.
    for _, i, j in edges:
        uf.union(i, j)

    # Group all nodes by their root.
    components = {}
    # Make sure to consider all nodes in edges.
    for _, i, j in edges:
        root_i = uf.find(i)
        root_j = uf.find(j)
        components.setdefault(root_i, set()).update([i, j])
        # The above also ensures both i and j are in the same component.

    return list(components.values())


import os
import csv

def save_graph_csv(edges: list, name):
    """
    Expects edges to be a list of tuples (w, i, j), where w is the edge weight, and i and j are the node IDs
    Saves the edges list as a CSV file with the given name under filtered_Data/csv_files/
    """
    # Define the directory where the CSV files will be saved
    output_dir = "filtered_Data/csv_files"
    
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the full path for the CSV file
    csv_file_path = os.path.join(output_dir, f"{name}.csv")
    
    # Write the edges to a CSV file
    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row
        writer.writerow(['Weight', 'Source', 'Target'])
        
        # Write each edge as a row in the CSV file
        for w, i, j in edges:
            writer.writerow([w, i, j])

    print(f"CSV file saved to {csv_file_path}")