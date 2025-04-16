import torch


def disparity_epe(prediction, reference, max_disparity=None):
    assert prediction.size() == reference.size()
    assert len(prediction.size())==3
    # find all valid gt values
    
    if max_disparity:
        valid_mask = (reference>0) & (reference < max_disparity)
    else:
        valid_mask = reference>0
        
    diff = torch.abs(prediction - reference)
    diff[~valid_mask] =0
    valid_pixels = torch.sum(valid_mask)
    err = torch.sum(diff)
    batch_error = err/valid_pixels
    #regect samples with no disparity values out of range.
    return torch.mean(batch_error[valid_pixels>0]).detach()

# def disparity_bad3(prediction, reference, max_disparity=None):
#     # import ipdb; ipdb.set_trace()
#     assert prediction.size() == reference.size()
#     assert len(prediction.size())==3
#     # find all valid gt values
    
#     if max_disparity:
#         valid_mask = (reference>0) & (reference < max_disparity)
#     else:
#         valid_mask = reference>0
        
#     diff = torch.abs(prediction - reference)
#     diff[~valid_mask]=0
#     bad_px = diff>3
#     valid_pixels = torch.sum(valid_mask)
#     bad_px = torch.sum(bad_px)

#     err = (bad_px/valid_pixels)*100
#     return err.mean().detach()

def disparity_bad3(D_est, D_gt,  max_disparity=None, thres=3):
    assert isinstance(thres, (int, float))
    assert D_est.size() == D_gt.size()
    assert len(D_est.size())==3

    if max_disparity:
        valid_mask = (D_gt>0) & (D_gt < max_disparity)
    else:
        valid_mask = D_gt>0

    D_est, D_gt = D_est[valid_mask], D_gt[valid_mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())