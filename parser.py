import argparse
import statistics

def parse_simulation_csv_file(filename):
    """
    Reads and parses the CSV file into a list of blocks.
    Each block includes:
      - Simulation parameters as a dictionary (mapping parameter names to float values)
      - The original simulation parameter header and values (as strings)
      - Voter token percentages for I/E, I/U, U/E, U/U
      - An Accuracy value
    """
    with open(filename, 'r') as f:
        # Read all lines and remove empty lines
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("Simulation Parameters:"):
            block = {}
            i += 1  # Skip the "Simulation Parameters:" line
            
            # Get simulation parameters header and values
            sim_params_header = lines[i]
            i += 1
            sim_params_values = lines[i]
            i += 1

            # Convert the simulation parameters into a dictionary
            keys = [key.strip() for key in sim_params_header.split(',')]
            values = [float(val.strip()) for val in sim_params_values.split(',')]
            sim_params = dict(zip(keys, values))
            
            block["sim_params"] = sim_params
            block["sim_params_header"] = sim_params_header
            block["sim_params_values"] = sim_params_values

            # Look for the voter token percentages
            if i < len(lines) and lines[i].startswith("Voter Type"):
                i += 1  # Skip header "Voter Type,Percent of Total Average Tokens"
                voter_data = {}
                # Expect 4 lines for I/E, I/U, U/E, U/U
                for _ in range(4):
                    voter_line = lines[i]
                    voter, percent = voter_line.split(',')
                    voter_data[voter] = float(percent.replace('%', '').strip())
                    i += 1
                block["voter_data"] = voter_data

                # Next line is the Accuracy line
                if i < len(lines) and lines[i].startswith("Accuracy"):
                    _, acc_percent = lines[i].split(',')
                    block["accuracy"] = float(acc_percent.replace('%', '').strip())
                    i += 1
            blocks.append(block)
        else:
            i += 1
    return blocks

def filter_blocks(blocks):
    """
    Filters blocks to include only those entries where:
      1. Accuracy is above 90%
      2. I/E > I/U > U/E > U/U
      3. prob_vote_correct_informed (simulation parameter) is between 0.6 and 0.7
      4. prob_vote_engaged >= prob_vote_disengaged
      5. prob_vote_correct_informed >= prob_vote_correct_uninformed
      6. prob_obj_correct_informed >= prob_obj_correct_uninformed
    """
    filtered = []
    for block in blocks:
        accuracy = block.get("accuracy", 0)
        voter = block.get("voter_data", {})
        ie = voter.get("I/E", 0)
        iu = voter.get("I/U", 0)
        ue = voter.get("U/E", 0)
        uu = voter.get("U/U", 0)
        sim_params = block.get("sim_params", {})
        prob_vote_engaged = sim_params.get("prob_vote_engaged", 0)
        prob_vote_disengaged = sim_params.get("prob_vote_disengaged", 0)
        prob_vote_correct_informed = sim_params.get("prob_vote_correct_informed", 0)
        prob_vote_correct_uninformed = sim_params.get("prob_vote_correct_uninformed", 0)
        prob_obj_correct_informed = sim_params.get("prob_obj_correct_informed", 0)
        prob_obj_correct_uninformed = sim_params.get("prob_obj_correct_uninformed", 0)
        
        if (accuracy > 90 and 
            (ie > iu) and (iu > ue) and (ue > uu) and 
            #0.9 <= prob_vote_correct_informed < 1.0 and 
            #prob_vote_correct_informed == 1.0 and
            prob_vote_engaged >= prob_vote_disengaged and
            prob_vote_correct_informed >= prob_vote_correct_uninformed and
            prob_obj_correct_informed >= prob_obj_correct_uninformed):
            filtered.append(block)
    return filtered

def sort_blocks(blocks, sort_param, ascending=True):
    """
    Sorts blocks based on the simulation parameter specified by sort_param.
    Supports both direct and computed parameters.
    """
    def get_sort_value(block, sort_param):
        sim_params = block.get("sim_params", {})
        # If the parameter exists directly, return it.
        if sort_param in sim_params:
            return float(sim_params[sort_param])
        # Compute the parameter if it's one of the computed ones.
        if sort_param == "prob_object_correct":
            return float(sim_params.get("prob_obj_correct_informed", 0) + sim_params.get("prob_obj_correct_uninformed", 0))
        elif sort_param == "prob_vote":
            return float(sim_params.get("prob_vote_engaged", 0) + sim_params.get("prob_vote_disengaged", 0))
        elif sort_param == "prob_vote_correct":
            return float(sim_params.get("prob_vote_correct_informed", 0) + sim_params.get("prob_vote_correct_uninformed", 0))
        return 0

    return sorted(
        blocks, 
        key=lambda block: get_sort_value(block, sort_param), 
        reverse=not ascending
    )

def write_filtered_blocks(blocks, output_filename):
    """
    Writes each block into the output CSV file in the same format,
    with a blank line separating each block.
    
    The output CSV block has:
      - The original simulation parameters header and values.
      - A blank line.
      - A new header line for computed parameters.
      - A new row with computed values:
            prob_vote = prob_vote_engaged + prob_vote_disengaged
            prob_vote_correct = prob_vote_correct_informed + prob_vote_correct_uninformed
            prob_object_correct = prob_obj_correct_informed + prob_obj_correct_uninformed
      - Voter token percentages and Accuracy.
      
    At the end, the function appends a final section with aggregated statistics for
    prob_vote and prob_object_correct over all filtered simulations.
    """
    # Lists to collect computed values from each simulation
    prob_vote_correct_list = []
    prob_vote_list = []
    prob_obj_correct_list = []
    
    with open(output_filename, 'w') as f:
        for block in blocks:
            # Write simulation parameters block
            f.write("Simulation Parameters:\n")
            f.write(block["sim_params_header"] + "\n")
            f.write(block["sim_params_values"] + "\n")
            f.write("\n")
            
            # Compute the new parameters from simulation parameters dictionary
            sim_params = block.get("sim_params", {})
            prob_vote = sim_params.get("prob_vote_engaged", 0) + sim_params.get("prob_vote_disengaged", 0)
            prob_vote_correct = sim_params.get("prob_vote_correct_informed", 0) + sim_params.get("prob_vote_correct_uninformed", 0)
            prob_object_correct = sim_params.get("prob_obj_correct_informed", 0) + sim_params.get("prob_obj_correct_uninformed", 0)
            
            # Collect computed values for later aggregation
            prob_vote_correct_list.append(prob_vote_correct)
            prob_vote_list.append(prob_vote)
            prob_obj_correct_list.append(prob_object_correct)
            
            # Write the computed parameters header and values
            f.write("prob_vote, prob_vote_correct, prob_object_correct\n")
            f.write(f"{prob_vote},{prob_vote_correct},{prob_object_correct}\n")
            
            # Write voter data and accuracy
            voter = block["voter_data"]
            f.write("I/E," + f"{voter.get('I/E', 0):.2f}%" + "\n")
            f.write("I/U," + f"{voter.get('I/U', 0):.2f}%" + "\n")
            f.write("U/E," + f"{voter.get('U/E', 0):.2f}%" + "\n")
            f.write("U/U," + f"{voter.get('U/U', 0):.2f}%" + "\n")
            f.write("Accuracy," + f"{block.get('accuracy', 0):.2f}%" + "\n")
            f.write("\n")  # Blank line to separate blocks
        
        # After writing all blocks, if we have any, compute aggregated stats.
        if prob_vote_list and prob_obj_correct_list:
            pv_min = min(prob_vote_list)
            pv_avg = statistics.mean(prob_vote_list)
            pv_std = statistics.pstdev(prob_vote_list)  # population standard deviation
            
            poc_min = min(prob_obj_correct_list)
            poc_avg = statistics.mean(prob_obj_correct_list)
            poc_std = statistics.pstdev(prob_obj_correct_list)

            pvc_min = min(prob_vote_correct_list)
            pvc_avg = statistics.mean(prob_vote_correct_list)
            pvc_std = statistics.pstdev(prob_vote_correct_list)
            
            # Write the aggregated statistics section
            f.write("Statistics:\n")
            f.write("Prob_vote_min,Prob_vote_avg,Prob_vote_std\n")
            f.write(f"{pv_min,pv_avg,pv_std}\n")
            f.write("Prob_object_correct_min,Prob_object_correct_avg,Prob_object_correct_std\n")
            f.write(f"{poc_min,poc_avg,poc_std}\n")
            f.write("Prob_vote_correct_min,Prob_vote_correct_avg,Prob_vote_correct_std\n")
            f.write(f"{pvc_min,pvc_avg,pvc_std}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse, filter, and sort simulation CSV blocks.")
    parser.add_argument("--input", type=str, default="output.csv", help="Input CSV file")
    parser.add_argument("--output", type=str, default="parser_output.csv", help="Output CSV file for filtered blocks")
    parser.add_argument("--sort_param", type=str, default=None,
                        help="Simulation parameter to sort by (e.g., prob_obj_correct_uninformed, prob_object_correct, prob_vote)")
    parser.add_argument("--order", type=str, choices=["asc", "desc"], default="asc",
                        help="Sort order: 'asc' for ascending (lowest first), 'desc' for descending (highest first)")

    args = parser.parse_args()
    
    # Parse the input file
    blocks = parse_simulation_csv_file(args.input)
    # Filter blocks based on the specified criteria
    filtered_blocks = filter_blocks(blocks)
    
    # Show how many entries with specified conditions were found
    count_filtered = len(filtered_blocks)
    print(f"Found {count_filtered} entries meeting the specified conditions.")
    
    # Sort the blocks if a sort parameter is provided
    if args.sort_param:
        ascending = args.order == "asc"
        filtered_blocks = sort_blocks(filtered_blocks, args.sort_param, ascending)
    
    # Write the filtered (and sorted) blocks to the output file
    write_filtered_blocks(filtered_blocks, args.output)
    
    print(f"Filtered and sorted blocks written to {args.output}.")
