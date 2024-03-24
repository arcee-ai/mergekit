import torch
import scipy
import sys
import numpy as np
import pdb
from collections import defaultdict

#####################################################################################################################################
############################################################## HELPERS ##############################################################
#####################################################################################################################################

def remove_col(x, idx):
    return torch.cat([x[:, :idx], x[:, idx+1:]], dim=-1)

def compute_correlation(covariance, eps=1e-7):
    covariance = torch.nan_to_num(covariance)
    std = torch.diagonal(covariance).sqrt() # there can be some infs in the covariance matrix
    covariance = covariance / (torch.clamp(torch.nan_to_num(torch.outer(std, std)),min=eps))
    return covariance


#####################################################################################################################################
#################################################### MATCHING/ALIGNMENT FUNCTIONS ###################################################
#####################################################################################################################################


def match_tensors_permute(r=.5, get_merge_value=False, 
                          print_costs=False, no_absval=False,
                          correlation_matrix=None):
    """
    This function is adapted from ZipIt! (https://github.com/gstoica27/ZipIt)

    Matches arbitrary models by permuting all to the spaces of the first in your graph list. 
    Mimics Rebasin methods. 
    """

    correlation = correlation_matrix

    O = correlation.shape[0]
    N = int(1/(1 - r) + 0.5)
    Om = O // N
    device = correlation.device
    
    mats = [torch.eye(Om, device=device)]
    cost = 0
    for i in range(1, N):
        try:
            corr_matrix = correlation[:Om, Om*i:Om*(i+1)].cpu().numpy()
            if no_absval == False:
                corr_matrix = np.absolute(corr_matrix)
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(
                corr_matrix, maximize=True)
            cost =  corr_matrix[row_ind, col_ind].sum()
            # correlation subset is is [0:4096, 4096:8192]
            # correlation between the first graph's and second graph's features
        except:
            pdb.set_trace()

        new_mat = torch.eye(Om, device=device)[torch.tensor(col_ind).long().to(device)]
        mats.append(new_mat.T)

    unmerge_mats = mats
        
    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    if get_merge_value:
        merge_value = correlation[:Om, Om*i:Om*(i+1)].cpu().numpy()[row_ind, col_ind].mean()
        return merge.T, unmerge, merge_value
    if print_costs:
        cost = cost / merge.shape[0]
        print(f'cost: {cost}')

    return merge.T, unmerge, None, cost / merge.shape[0]

def match_tensors_permute_MHA(n_heads, permute_heads=False, 
                              head_assignments=[], r=.5, get_merge_value=False, 
                              print_costs=False, no_absval=False, 
                              correlation_matrix=None):
    """
    Handles different head permutations in attention
    """
    correlation = correlation_matrix

    O = correlation.shape[0]

    N = int(1/(1 - r) + 0.5) # num models
    Om = O // N # matrix dimension
    device = correlation.device
    query_size = Om // n_heads 
    
    mats = [torch.eye(Om, device=device)]
    head_perms = []

    # compute head perms in order
    if permute_heads == False:
        cost = 0
        for i in range(1, N): #just once if 2 models]
            for j in range(n_heads):
                try:
                    # by head
                    corr_submatrix =  correlation[query_size * j:query_size * (j+1), Om*i + query_size*j:Om*i + query_size*(j+1)].cpu().numpy()
                    if no_absval == False:
                        corr_submatrix = np.absolute(corr_submatrix)
                    row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_submatrix, maximize=True)


                    head_perms.append(torch.tensor(col_ind + j*query_size))
                    cost += corr_submatrix[row_ind, col_ind].sum()
                    
                    # for whole model correlation subset is is [0:4096, 4096:8192]
                    # correlation between the first graph's and second graph's features
                except:
                    pdb.set_trace()
        outer_col_ind = np.arange(n_heads)
    # compute head perms out of order according to predefined ordering or find our own
    elif permute_heads == True:
        cost = 0
        col_inds_storage = defaultdict(lambda: defaultdict(int))
        if head_assignments != []:
            outer_row_ind = np.arange(n_heads)
            outer_col_ind = head_assignments
            for i in range(n_heads):
                head1_idx = [query_size * outer_row_ind[i], query_size * (outer_row_ind[i] + 1)]
                head2_idx = [Om + query_size * outer_col_ind[i], Om + query_size * (outer_col_ind[i] + 1)]
                # take abs value of submatrix of correlations
                corr_submatrix = correlation[head1_idx[0]:head1_idx[1], head2_idx[0]:head2_idx[1]].cpu().numpy()
                if no_absval == False:
                    corr_submatrix = np.absolute(corr_submatrix)
                # compute perm for head j & head k 
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_submatrix, maximize=True)

                cost += corr_submatrix[row_ind, col_ind].sum()
                col_inds_storage[outer_row_ind[i]][outer_col_ind[i]] = col_ind
           
        else: 
            costs = np.ones((n_heads, n_heads)) * -sys.maxsize  # cost matrix for hungarian algo steps
            for i in range(1, N):  #just once if 2 models 
                for j in range(n_heads): # outer loop through all heads
                    for k in range(n_heads):  # inner loop through heads >= current head j
                        head1_idx = [query_size * j, query_size * (j+1)]
                        head2_idx = [Om * i + query_size * k, Om * i + query_size * (k+1)]

                        # take abs value of submatrix of correlations
                        corr_submatrix = correlation[head1_idx[0]:head1_idx[1], head2_idx[0]:head2_idx[1]].cpu().numpy()
                        if no_absval == False:
                            corr_submatrix = np.absolute(corr_submatrix)

                        # compute perm for head j & head k 
                        row_ind, col_ind = scipy.optimize.linear_sum_assignment(corr_submatrix, maximize=True)

                        # store cost (cost is maximized here)
                        costs[j,k] = corr_submatrix[row_ind, col_ind].sum()
                        #costs[k,j] = costs[j,k] # make symmetric

                        # store perm so we don't have to recompute it later
                        col_inds_storage[j][k] = col_ind


            outer_row_ind, outer_col_ind = scipy.optimize.linear_sum_assignment(costs, maximize=True) # get assignment with lowest cost
            cost += costs[outer_row_ind, outer_col_ind].sum()

        for j in range(n_heads):
            head_1 = outer_row_ind[j] # these are in order, outer_row_ind[j] = j
            head_2 = outer_col_ind[j]

            head_perm = col_inds_storage[head_1][head_2]
            head_perms.append(torch.tensor(head_perm + query_size*head_2))

    new_mat = torch.eye(Om, device=device)[torch.tensor(torch.cat(head_perms)).long().to(device)]
    mats.append(new_mat.T)
    
    unmerge_mats = mats
    
    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    if print_costs:
        cost = cost / merge.shape[0]
        print(f'cost: {cost}')
    if get_merge_value:
        merge_value = correlation[:Om, Om*i:Om*(i+1)].cpu().numpy()[row_ind, col_ind].mean()
        return merge.T, unmerge, merge_value
    return merge.T, unmerge, outer_col_ind, cost / merge.shape[0]

def match_tensors_identity(r=.5, correlation_matrix=None, **kwargs):
    # weight averaging.  
    
    correlation = correlation_matrix
    O = correlation.shape[0]

    N = int(1/(1 - r) + 0.5)
    Om = O // N
    device = correlation.device
    corr_matrix = correlation[:Om, Om:Om*2].cpu().numpy()
    cost = corr_matrix.trace()

    mats = [torch.eye(Om, device=device) for _ in range(N)]
    
    unmerge_mats = mats

    unmerge = torch.cat(unmerge_mats, dim=0)
    merge = torch.cat(mats, dim=0)
    merge = merge / (merge.sum(dim=0, keepdim=True) + 1e-5)
    cost = cost / merge.shape[0]
    return merge.T, unmerge, None, cost

