import numpy as np

def mmr(click_prob_list, similarity_matrix, rec_num, lambda_constant=0.5): # O(n^2)
    candidate_num = len(click_prob_list)
    sel_idx_list, candidate_idx_set = [], set([i for i in range(candidate_num)])

    while len(candidate_idx_set) > 0:
        highest_score = None
        new_select_idx = None

        for idx in candidate_idx_set:
            part1 = click_prob_list[idx]

            part2 = None
            for sel_idx in sel_idx_list:
                sim_score = similarity_matrix[idx][sel_idx]
                if part2 == None or sim_score > part2:
                    part2 = sim_score
            if part2 == None: part2=0

            score = lambda_constant * (part1 - (1-lambda_constant)*part2)
            if highest_score == None or score > highest_score:
                highest_score = score
                new_select_idx = idx

        candidate_idx_set.remove(new_select_idx)
        sel_idx_list.append(new_select_idx)

        if len(sel_idx_list) >= rec_num:
            break

    return sel_idx_list

def mmr_new(click_prob_list, user_item_sim, rec_num, lambda_constant=0.5):

    score_list = [prob-(1-lambda_constant)*sim for prob, sim in zip(click_prob_list, user_item_sim)]
    order = np.argsort(score_list)[::-1]  # idx, big to small by score
    rec_idx_list = order[:rec_num]

    return rec_idx_list