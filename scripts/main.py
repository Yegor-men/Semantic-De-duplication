import sqlite3
import re
import os

import torch
import pandas as pd
from scripts import functions as F

from timeit import default_timer as time
from tqdm import tqdm

start_global = time()

# Create the database if it does not exist already
conn = sqlite3.connect("entries.db")
F.create_table(conn)

# Read the input file, select the last sheet and split it by Test Type
file_location = "data/Group_Payments-TD_END_TO_END_plus_SIT.xlsx"
sheet_names = pd.ExcelFile(file_location).sheet_names
df = pd.read_excel(file_location, sheet_name=sheet_names[-1])
df.columns = df.columns.str.strip()
unique_test_types = df["Test Type"].unique()

# Create a new file where each sheet is for a separate Test Type
output_folder = "filtered_Data"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "grouped_db.xlsx")
with pd.ExcelWriter(output_file) as writer:
    for test_type in unique_test_types:
        df_filtered = df[df["Test Type"] == test_type]
        replaceables = r'[\/\\\*\[\]\:\?]'
        sheet_name = re.sub(replaceables, '_', str(test_type))[:31]
        df_filtered.to_excel(writer, sheet_name=sheet_name, index=False)

grouped_file_path = "filtered_Data/grouped_db.xlsx"
all_sheets = pd.read_excel(grouped_file_path, sheet_name=None)

base_name, ext = os.path.splitext(grouped_file_path)
duplicate_file_path = f"{base_name}_DUPLICATE{ext}"

# Hyperparameters for the code: the model name, how big the matrices to multiply and the threshold to pass as duplicate
EMBEDDING_MODEL = "snowflake-arctic-embed2:568m-l-fp16"
BLOCK_SIZE = 4096
DUPLICATE_THRESHOLD = 0.99999

# Creates the duplicates file, and iterating over the sheets writes the sets of duplicates there
with pd.ExcelWriter(duplicate_file_path) as writer:
    for sheet_name, df_sheet in all_sheets.items():
        test_type = df_sheet["Test Type"].iloc[0]
        print(f"\nProcessing: {test_type}")

        # Converts each of the columns into a list
        id_list = df_sheet["Id"].tolist()
        name_list = df_sheet["Name"].tolist()
        description_list = df_sheet["Description"].fillna("").tolist()
        test_steps_list = df_sheet["Test Steps"].fillna("").tolist()

        # Creates a dictionary mapping the index in the list to the actual ID of the test case
        local_to_global_id_dict = {local_id: global_id for local_id, global_id in enumerate(id_list)}
        global_to_local_id_dict = {global_id: local_id for local_id, global_id in enumerate(id_list)}

        # Iterates over each test case and writes/updates it in the database if necessary
        progress_bar = tqdm(
            zip(id_list, name_list, description_list, test_steps_list),
            total=len(id_list),
            desc=f"\tEncoding {len(id_list)} TCs"
        )

        for id, name, description, test_steps in progress_bar:
            existing_entry = F.get_entry_by_id(conn, id)
            formatted_test_case = F.format_test_case(name, description, test_steps)
            if existing_entry is None:
                F.add_entry(conn, id, formatted_test_case, EMBEDDING_MODEL)
            else:
                is_different = F.is_entry_different(existing_entry, formatted_test_case, EMBEDDING_MODEL)
                if is_different:
                    F.update_entry(conn, id, formatted_test_case, EMBEDDING_MODEL)
                    print(f"AJSDFLKAJSDFLKJASD;LKFJASD;LKFJASL;KD")

        # Retrieves the necessary embeddings for the IDs in the input file
        start_retrieve = time()
        embeddings = F.get_embeddings_by_ids(conn, id_list)
        end_retrieve = time()
        print(f"\tRetrieval: {end_retrieve - start_retrieve:.2f}s")

        # Turns the list of embeddings into one tensor with size [num elements, embedding dimension]
        ids, embedding_list = zip(*embeddings)
        embedding_matrix = torch.stack(embedding_list, dim=0)

        # Does matrix multiplication by blocks and returns a list of tuples (s, i, j) for the similar pairs
        filtered_list = F.get_filtered_similarity_matrix(embedding_matrix, BLOCK_SIZE, DUPLICATE_THRESHOLD)

        # Turns the indices for coordinates i and j into their ID values
        curated_list = [(s, local_to_global_id_dict[i], local_to_global_id_dict[j]) for (s, i, j) in filtered_list]
        end_matmul = time()
        print(f"\tMatrix multiplication: {end_matmul - end_retrieve:.2f}s")

        # Uses transitive closure based on if A -> B and B -> C, then A -> C, hence creating sets of duplicates
        similar_items = F.transitive_closure(curated_list)
        end_transitive_closure = time()
        print(f"\tTransitive closure: {end_transitive_closure - end_matmul:.2f}s")

        # Creates a separate CSV file for all the edges to be imported and used in Gephi
        F.save_graph_csv(curated_list, sheet_name)

        # Turns the list of sets of duplicates into what is written to the output xlsx file
        expanded_data = [list(item_set) for item_set in similar_items]
        duplicates_df = pd.DataFrame(expanded_data)
        duplicates_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        end_write = time()
        print(f"\tWrite duplicates: {end_write - end_transitive_closure:.2f}s", end="\n\n")

end_global = time()
print(f"Finished in {end_global - start_global:.2f} seconds")
