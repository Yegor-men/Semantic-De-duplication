import ollama
import torch
import sqlite3
import io

# Sets the device to be used to cuda if it is available, otherwise everything will be done on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


def create_table(conn):
    """
    Creates a .db for the test cases.
    Stores the ID, the test case, the model used for embedding and the embedding itself.
    """
    c = conn.cursor()
    c.execute(
        '''
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            test_case TEXT,
            model TEXT,
            embedding BLOB
        )
        '''
    )
    conn.commit()


def format_test_case(title: str, description: str, test_steps: str) -> str:
    """
    Formats a test case in accordance to a certain template. The output is a list with 1 element ready for Ollama.
    """
    return f"Title: {title}\n\nDescription: {description}\n\nTest Steps: {test_steps}"


def calculate_embedding(formatted_tc: str, model: str) -> torch.Tensor:
    """
    Calculates the embedding for a curated test case and returns the embedding for it.
    """
    input_text = [formatted_tc]
    responses = ollama.embed(model=model, input=input_text)
    embeddings = responses["embeddings"]
    embed_tensor = torch.tensor(embeddings).squeeze()
    return embed_tensor


def serialize_tensor(tensor):
    """
    Serializes the embedding tensor to be stored in the database.
    """
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def deserialize_tensor(blob):
    """
    Deserializes the embedding tensor to be retrieved from the database
    """
    buffer = io.BytesIO(blob)
    return torch.load(buffer)


def get_entry_by_id(conn, entry_id):
    """
    Returns everything about a test case based on the ID.
    """
    c = conn.cursor()
    c.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
    return c.fetchone()


def is_entry_different(existing_entry, new_test_case, new_model):
    """
    Checks if the existing test case entry is different from the proposed one.
    Returns either True or False.
    """
    id, test_case, model, blob = existing_entry
    return (test_case != new_test_case) or (model != new_model)


def add_entry(conn, id, formatted_tc, model):
    """
    Adds an entry to the database.
    """
    embedding = calculate_embedding(formatted_tc, model)
    embedding_blob = serialize_tensor(embedding)
    c = conn.cursor()
    c.execute(
        "INSERT INTO entries (id, test_case, model, embedding) VALUES (?, ?, ?, ?)",
        (id, formatted_tc, model, embedding_blob)
    )
    conn.commit()


def update_entry(conn, id, new_tc, new_model):
    """
    Updates an entry in the database.
    """
    embedding = calculate_embedding(new_tc, new_model)
    embedding_blob = serialize_tensor(embedding)
    c = conn.cursor()
    c.execute(
        "UPDATE entries SET test_case = ?, model = ?, embedding = ? WHERE id = ?",
        (new_tc, new_model, embedding_blob, id)
    )
    conn.commit()


def get_embeddings_by_ids(conn, id_list):
    """
    Returns a list of tuples (ID, embedding) based on a list of IDs
    """
    c = conn.cursor()
    placeholders = ', '.join('?' for _ in id_list)
    query = f"SELECT id, embedding FROM entries WHERE id IN ({placeholders})"
    c.execute(query, id_list)

    result_map = {}
    for row in c.fetchall():
        entry_id, blob = row
        embedding = deserialize_tensor(blob)
        result_map[entry_id] = embedding
    results = []
    for id in id_list:
        results.append((id, result_map[id]))
    return results


def get_filtered_similarity_matrix(embed_tensor, block_size, similarity_coefficient):
    """
    Does matrix multiplication by blocks and filters the results to return a list of tuples (s, i, j)
    which means that index i is similar to index j by a coefficient of s which is guaranteed to be over the
    similarity_coefficient.
    """
    # Hyperparameters for later
    num_embeds = embed_tensor.size(0)
    embed_size = embed_tensor.size(1)

    # Calculates the necessary number of blocks to cover the number of elements
    num_blocks = num_embeds // block_size + (1 if num_embeds % block_size else 0)

    filtered_list = []

    for i in range(num_blocks):
        i_start = i * block_size
        i_end = min((i + 1) * block_size, num_embeds)

        for j in range(num_blocks):
            j_start = j * block_size
            j_end = min((j + 1) * block_size, num_embeds)

            # Creates a new tensor based on the current coordinates to calculate stuff by blocks
            i_block = embed_tensor[i_start:i_end, :].to(device)
            j_block = embed_tensor[j_start:j_end, :].to(device)

            # Does matrix multiplication such that coordinates ij show how similar i is to j
            matrix = torch.matmul(i_block, j_block.T)

            # Filters the matrix out so that all values less than the similarity coefficient get turned to 0
            matrix[matrix < similarity_coefficient] = 0

            # Creates a mask to remove duplicates and self references by masking the lower triangle of the global matrix
            global_rows = torch.arange(i_start, i_end, device=device).unsqueeze(1)  # shape (block_rows, 1)
            global_cols = torch.arange(j_start, j_end, device=device).unsqueeze(0)  # shape (1, block_cols)
            upper_mask = global_rows < global_cols

            # Keeps only the values that fulfill the upper triangle and over the similarity coefficient
            valid_mask = (matrix != 0) & upper_mask

            # Retrieve indices and values where valid_mask is True
            nonzero_indices = torch.nonzero(valid_mask, as_tuple=False)
            scores = matrix[valid_mask]

            # Iterates over the indices and turns them to coordinates
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
    """
    Uses transitive closure to return a list of sets of duplicates.
    """
    uf = UnionFind()
    # Process each edge to union the nodes
    for _, i, j in edges:
        uf.union(i, j)

    # Group all nodes by their root
    components = {}
    # Make sure to consider all nodes in edges
    for _, i, j in edges:
        root_i = uf.find(i)
        root_j = uf.find(j)
        components.setdefault(root_i, set()).update([i, j])
        # The above also ensures both i and j are in the same component

    return list(components.values())


import os
import csv


def save_graph_csv(edges: list, name: str):
    """
    Saves a CSV file of the edges to be used for Gephi.
    """
    output_dir = "filtered_Data/csv_files"
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, f"{name}.csv")

    with open(csv_file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Weight', 'Source', 'Target'])

        for w, i, j in edges:
            writer.writerow([w, i, j])

    print(f"\tCSV edges file for Gephi saved")
