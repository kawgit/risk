import json

def print_java_const_liner(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    territories = data.get("territories", {})
    name_to_id = {attr["name"]: attr["id"] for attr in territories.values()}
    size = max(name_to_id.values()) + 1
    
    # Initialize and fill matrix
    matrix = [[0 for _ in range(size)] for _ in range(size)]
    for t_attr in territories.values():
        i = t_attr["id"]
        matrix[i][i] = 1  # i -> i
        for n_name in t_attr.get("neighbours", []):
            if n_name in name_to_id:
                matrix[i][name_to_id[n_name]] = 1

    # Format into a Java array literal
    # Example: {{1,1,0}, {1,1,1}, {0,1,1}}
    rows = [ "{" + ",".join(map(str, row)) + "}" for row in matrix ]
    java_literal = "{" + ",".join(rows) + "}"
    
    print(f"public static final int[][] ADJACENCY_MATRIX = {java_literal};")

# Run it
print_java_const_liner('map.json')