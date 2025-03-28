import torch
import pandas as pd
import re
import os
from scripts import functions as F
import importlib

importlib.reload(F)
from timeit import default_timer as time
import sqlite3
from tqdm import tqdm

start = time()

conn = sqlite3.connect("entries.db")
F.create_table(conn)

file_location = "data/Group_Payments-TD_END_TO_END_plus_SIT.xlsx"
df = pd.read_excel(file_location, sheet_name=1)

df.columns = df.columns.str.strip()
unique_test_types = df["Test Type"].unique()

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

DUPLICATE_THRESHOLD = 0.99999
BLOCK_SIZE = 4096
embedding_model = "snowflake-arctic-embed2:568m-l-fp16"

with pd.ExcelWriter(duplicate_file_path) as writer:
    for sheet_name, df_sheet in all_sheets.items():
        test_type = df_sheet["Test Type"].iloc[0]
        print(f"\nProcessing: {test_type}")

        id_list = df_sheet["Id"].tolist()
        name_list = df_sheet["Name"].tolist()
        description_list = df_sheet["Description"].fillna("").tolist()
        test_steps_list = df_sheet["Test Steps"].fillna("").tolist()

        local_to_global_id_dict = {local_id: global_id for local_id, global_id in enumerate(id_list)}
        global_to_local_id_dict = {global_id: local_id for local_id, global_id in enumerate(id_list)}

        print(f"Local size {len(local_to_global_id_dict)} Global {len(global_to_local_id_dict)}")

        start_encode = time()
        for id, name, description, test_steps in tqdm(zip(id_list, name_list, description_list, test_steps_list),
                                                      total=len(id_list), desc=f"\tEncoding {len(id_list)} TCs"):
            existing_entry = F.get_entry_by_id(conn, id)
            new_entry = (id, name, description, test_steps, embedding_model)
            if existing_entry is None:
                F.add_entry(conn, new_entry)
            else:
                is_different = F.is_entry_different(existing_entry, new_entry)
                if is_different:
                    F.update_entry(conn, new_entry)
        end_encode = time()
        print(f"\tEncoding: {end_encode - start_encode:.2f}s", end = " | ")

        embeddings = F.get_embeddings_by_ids(conn, id_list)
        end_retrieve = time()
        print(f"Retrieve: {end_retrieve - end_encode:.2f}s", end = " | ")

        _, embedding_list = zip(*embeddings)

        embedding_matrix = torch.stack(embedding_list, dim=0)

        #
        # meow1=global_to_local_id_dict['TC-297120']
        # meow2=global_to_local_id_dict['TC-293775']
        #
        # e1 = F.get_embeddings_by_ids(conn, ['TC-297120'])[0][1]
        # # _, ee1 = zip(*e1)
        # e2 = F.get_embeddings_by_ids(conn, ['TC-293775'])[0][1]
        # # _, ee2 = zip(*e2)
        #
        # print(f"TC-297120: {e1}")
        # print(f"TC-293775: {e2}")
        #
        #
        # dproduct = torch.sum(e1 * e2)
        #
        # print(f"dproduct: {dproduct}")


        filtered_list = F.get_filtered_similarity_matrix(embedding_matrix, BLOCK_SIZE, DUPLICATE_THRESHOLD)


        # for (s, i, j) in filtered_list:
        #     if (i==meow1 and j==meow2) or (j==meow1 and i==meow2):
        #         print(f"MEOW {s} {i} {j}")

# MEOW 1.000000238418579 63866 64141
        curated_list = [(s, local_to_global_id_dict[i], local_to_global_id_dict[j]) for (s, i, j) in filtered_list]

        end_matmul = time()
        print(f"Matmul: {end_matmul - end_retrieve:.2f}s", end = " | ")

        similar_items = F.transitive_closure(curated_list)
        # print(similar_items)
        end_transitive_closure = time()
        print(f"Transitive closure: {end_transitive_closure - end_matmul:.2f}s", end = " | ")

        expanded_data = [list(item_set) for item_set in similar_items]

        duplicates_df = pd.DataFrame(expanded_data)
        duplicates_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
        end_write = time()
        print(f"Write: {end_write - end_transitive_closure:.2f}s", end = "\n\n")

        F.save_graph_csv(curated_list, sheet_name)

end = time()
total_time = end - start

print(f"Finished in {total_time:.2f} seconds")
