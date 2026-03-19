import os
import networkx as nx
from tqdm.auto import tqdm
import csv

# ==========================================
# ?숋툘 ?ㅼ젙 (寃쎈줈 ?뺤씤 ?꾩닔)
# ==========================================
# 1. ?낅젰 ?뚯씪: 諛⑷툑 sort 紐낅졊?대줈 ?뺣젹???뚯씪
INPUT_FILE = "./results/sorted_input.tsv" 

# 2. 異쒕젰 ?뚯씪
OUTPUT_FILE = "./results/submission_post_processed.tsv"

# 3. GO 援ъ“ ?뚯씪
OBO_FILE = "./data/raw/train/go-basic.obo" 
# ==========================================

def create_ontology_graph(obo_path):
    print("?㎥ Ontology Graph ?앹꽦 以?..")
    ontology_graph = nx.DiGraph()
    
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"?뚯씪???놁뒿?덈떎: {obo_path}")

    current_id = None
    with open(obo_path, "r") as file:
        for line in file:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: "):
                current_id = line.split("id: ")[1].strip()
                ontology_graph.add_node(current_id)
            elif line.startswith("is_a: "):
                parent_id = line.split()[1].strip()
                if current_id:
                    ontology_graph.add_edge(current_id, parent_id)
            elif line.startswith("relationship: part_of "):
                parts = line.split()
                if len(parts) >= 3:
                    parent_id = parts[2].strip()
                    if current_id:
                        ontology_graph.add_edge(current_id, parent_id)

    if not nx.is_directed_acyclic_graph(ontology_graph):
        for cycle in nx.simple_cycles(ontology_graph):
            ontology_graph.remove_edge(cycle[0], cycle[1])
    
    topological_order = list(nx.topological_sort(ontology_graph))
    child_to_parents = {node: list(ontology_graph.successors(node)) for node in ontology_graph.nodes()}
    
    return topological_order, child_to_parents

def process_single_protein(protein_id, buffer, topological_order, child_to_parents, writer):
    """
    ???⑤갚吏덉쓽 ?곗씠??buffer)瑜?諛쏆븘???꾪뙆(Propagation) ??利됱떆 ?뚯씪???
    """
    # 1. ?뺤뀛?덈━濡?蹂??
    term_scores = {term: score for term, score in buffer}
    
    # 2. ?먯닔 ?꾪뙆 (True Path Rule)
    for child_term in topological_order:
        if child_term in term_scores:
            child_score = term_scores[child_term]
            for parent_term in child_to_parents.get(child_term, []):
                current_parent_score = term_scores.get(parent_term, 0.0)
                if child_score > current_parent_score:
                    term_scores[parent_term] = child_score
    
    # 3. 利됱떆 ???(硫붾え由??댁젣 ?④낵)
    for term, score in term_scores.items():
        if score >= 0.001: # ?⑸웾 理쒖쟻??
            writer.writerow([protein_id, term, f"{score:.4f}"])

def run_streaming_propagation():
    # 1. Ontology 濡쒕뱶
    ontology_order, parent_mapping = create_ontology_graph(OBO_FILE)
    
    print(f"?뙄 ?ㅽ듃由щ컢 泥섎━ ?쒖옉: {INPUT_FILE} -> {OUTPUT_FILE}")
    print("   (硫붾え由щ? ?꾨겮湲??꾪빐 ??以꾩뵫 ?쎌뼱??泥섎━?⑸땲??")
    
    # 2. ?뚯씪 ?ㅽ듃由щ컢
    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w', newline='') as f_out:
        reader = csv.reader(f_in, delimiter='\t')
        writer = csv.writer(f_out, delimiter='\t')
        
        current_pid = None
        buffer = [] # ???⑤갚吏덉쓽 ?뺣낫留??댁쓣 ?꾩떆 洹몃쫯
        
        # 吏꾪뻾瑜??쒖떆瑜??꾪빐 ??듭쟻??以???移댁슫??(?좏깮?ы빆)
        # pbar = tqdm(reader) 
        
        for row in tqdm(reader, desc="Processing"):
            if not row: continue
            
            try:
                pid, term, score = row[0], row[1], float(row[2])
            except ValueError:
                continue # ?ㅻ뜑???댁긽??以??ㅽ궢
            
            # ?⑤갚吏?ID媛 諛붾뚮㈃ -> 吏湲덇퉴吏 紐⑥?嫄?泥섎━?섍퀬 鍮꾩?
            if pid != current_pid:
                if current_pid is not None:
                    process_single_protein(current_pid, buffer, ontology_order, parent_mapping, writer)
                
                current_pid = pid
                buffer = [] # 踰꾪띁 珥덇린??(硫붾え由??댁젣)
            
            # ?꾩옱 ?⑤갚吏??뺣낫 紐⑥쑝湲?
            buffer.append((term, score))
        
        # 留덉?留??⑤갚吏?泥섎━
        if current_pid is not None and buffer:
            process_single_protein(current_pid, buffer, ontology_order, parent_mapping, writer)

    print("???꾨즺! 硫붾え由?遺議??놁씠 ?앸궗?듬땲??")

if __name__ == "__main__":
    run_streaming_propagation()
