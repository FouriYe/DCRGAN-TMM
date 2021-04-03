import torch
import torch.nn.functional as F
import numpy as np
import sklearn


def pairwise_mahalanobis(S1, S2, Cov_1=None):
    """
        S1: C1 x K matrix (torch.FloatTensor)
          -> C1 K-dimensional semantic features
        S2: C2 x K matrix (torch.FloatTensor)
          -> C2 K-dimensional semantic features
        Sigma_1: K x K matrix (torch.FloatTensor)
          -> inverse of the covariance matrix Sigma; used to compute Mahalanobis distances
          by default Sigma is the identity matrix (and so distances are euclidean distances)
        
        returns an C1 x C2 matrix corresponding to the Mahalanobis distance between each element of S1 and S2
        (Equation 5)
    """
    if S1.dim() != 2 or S2.dim() != 2 or S1.shape[1] != S2.shape[1]:
        raise RuntimeError("Bad input dimension")
    C1, K = S1.shape
    C2, K = S2.shape
    if Cov_1 is None:
        Cov_1 = torch.eye(K)
    if Cov_1.shape != (K, K):
        raise RuntimeError("Bad input dimension")
    
    S1S2t = S1.matmul(Cov_1).matmul(S2.t())
    S1S1 = S1.matmul(Cov_1).mul(S1).sum(dim=1, keepdim=True).expand(-1, C2)
    S2S2 = S2.matmul(Cov_1).mul(S2).sum(dim=1, keepdim=True).t().expand(C1, -1)
    return torch.sqrt(torch.abs(S1S1 + S2S2 - 2. * S1S2t) + 1e-32)  # to avoid numerical instabilities

def distance_matrix(S, mahalanobis=True, mean=1., std=0.5):
    """
        S: C x K matrix (numpy array)
          -> K-dimensional semantic features of C classes
        mahalanobis: indicates whether to use Mahalanobis distance (uses euclidean distance if False)
        mean & std: target mean and standard deviation
        
        returns a C x C matrix corresponding to the Mahalanobis distance between each pair of elements of S
        rescaled to have approximately target mean and standard deviation while keeping values positive
        (Equation 6)
    """
    Cov_1 = None
    if mahalanobis:
        Cov, _ = sklearn.covariance.ledoit_wolf(S) # robust estimation of covariance matrix
        Cov_1 = torch.FloatTensor(np.linalg.inv(Cov))
    S = torch.FloatTensor(S)
    
    distances = pairwise_mahalanobis(S, S, Cov_1)
    
    # Rescaling to have approximately target mean and standard deviation while keeping values positive
    max_zero_distance = distances.diag().max()
    positive_distances = np.array([x for x in distances.view(-1) if x > max_zero_distance])
    emp_std = float(positive_distances.std())
    emp_mean = float(positive_distances.mean())
    distances = F.relu(std * (distances - emp_mean) / emp_std + mean)
    emp_std = float(distances.std())
    emp_mean = float(distances.mean())
    distances = F.relu(std * (distances - emp_mean) / emp_std + mean)
    return distances

def _pairwise_distances(feature, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        feature: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = feature@torch.transpose(feature,1,0)

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.relu(distances)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0.0).type(torch.cuda.FloatTensor)
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.shape[0]).type(torch.cuda.ByteTensor)
    indices_not_equal = ~indices_equal
    i,j = indices_not_equal.shape
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j&i_not_equal_k)&j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    label_equal = label_equal.type(torch.cuda.ByteTensor)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = i_equal_j&(~i_equal_k)

    # Combine the two masks
    mask = distinct_indices&valid_labels

    return mask

def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.shape[0]).type(torch.cuda.ByteTensor)
    indices_not_equal = ~indices_equal

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0).eq(labels.unsqueeze(1))
    labels_equal = labels_equal.type(torch.cuda.ByteTensor)

    # Combine the two masks
    mask = indices_not_equal&labels_equal

    return mask

def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.

    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]

    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = labels.unsqueeze(0).eq(labels.unsqueeze(1))

    mask = ~labels_equal

    return mask

def batch_all_triplet_loss(feature, labels, margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        feature: tensor of shape (batch_size, embed_dim)
        labels: labels of the batch, of size (batch_size,)
        margin: margin for triplet loss

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = _pairwise_distances(feature, squared=squared)

    # shape (batch_size, batch_size, 1)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    # shape (batch_size, 1, batch_size)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
    # and the 2nd (batch_size, 1, batch_size)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    # print(triplet_loss.shape)

    # Put to zero the invalid triplets 
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    mask = _get_triplet_mask(labels)
    mask = mask.type(torch.cuda.FloatTensor)
    # print(mask.shape)
    triplet_loss = mask*triplet_loss

    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.relu(triplet_loss)

    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = torch.gt(triplet_loss,1e-16).type(torch.cuda.FloatTensor)
    num_positive_triplets = torch.sum(valid_triplets)
    num_valid_triplets = torch.sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def batch_hard_triplet_loss(feature, labels , margin, squared=False):
    """Build the triplet loss over a batch of embeddings.

    For each anchor, we get the hardest positive and hardest negative to form a triplet.

    Args:
        feature: tensor of shape (batch_size, embed_dim)
        labels: labels of the batch, of size (batch_size,)
        margin: margin for triplet loss

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(feature, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = mask_anchor_positive.type(torch.cuda.FloatTensor)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = mask_anchor_positive*pairwise_dist

    # shape (batch_size, 1)torch.max(a, dim=1, keepdim=True)
    hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]
    # print("hardest_positive_dist", torch.mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = mask_anchor_negative.type(torch.cuda.FloatTensor)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist = torch.max(pairwise_dist, dim=1, keepdim=True)[0]
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]
    # print("hardest_negative_dist", torch.mean(hardest_negative_dist))

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.relu(hardest_positive_dist - hardest_negative_dist + margin)

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss