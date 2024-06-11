import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def spherical2cartesial(x):
    output = torch.zeros(x.size(0), 3).to(x)
    output[:, 2] = -torch.cos(x[:, 1]) * torch.cos(x[:, 0])
    output[:, 0] = torch.cos(x[:, 1]) * torch.sin(x[:, 0])
    output[:, 1] = torch.sin(x[:, 1])

    return output

def cartesial2spherical(x):
    output = torch.zeros(x.size(0), 2).to(x)
    assert x.size(1) == 3
    x_norm = torch.nn.functional.normalize(x, p=2, dim=1, eps=1e-7).to(x)
    output[:, 0] = torch.atan2(x_norm[:, 0], -x_norm[:, 2])
    output[:, 1] = torch.asin(x_norm[:, 1])
    return output


def compute_angular_error(input, target):
    input_cart = spherical2cartesial(input)
    target_cart = spherical2cartesial(target)

    input_cart = input_cart.view(-1, 3, 1)
    target_cart = target_cart.view(-1, 1, 3)
    output_dot = torch.bmm(target_cart, input_cart)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = 180 / math.pi * torch.sum(output_dot)

    return output_dot

def compute_angular_error_cartesian(input, target):
    input_cart = torch.nn.functional.normalize(input, p=2, dim=1)
    target_cart = torch.nn.functional.normalize(target, p=2, dim=1)

    input_cart = input_cart.view(-1, 3, 1)
    target_cart = target_cart.view(-1, 1, 3)
    output_dot = torch.bmm(target_cart, input_cart)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = 180 / math.pi * torch.sum(output_dot)

    return output_dot


def get_events(y, marge=7):
    """
    Get events from a given prediction per label exclude none"""
    events = []
    current_event = {"start": None, "end": None, "label": None}

    for i in range(len(y)):
        if y[i] != 0 and current_event["start"] is None:
            current_event["start"] = i
            current_event["label"] = y[i]
        elif y[i] != current_event["label"] and current_event["start"] is not None:
            current_event["end"] = i - 1
            events.append(current_event.copy())
            current_event = {"start": None, "end": None, "label": None}

    if current_event["start"] is not None:
        current_event["end"] = len(y) - 1
        events.append(current_event.copy())

    # remove small events
    if (marge is not None) and (marge > 0):
        for e in events:
            if (e["end"] - e["start"]) <= marge:
                events.remove(e)

    return events


def smooth_pred(pred, marge=5):
    """
    Smooth the segmentation prediction using majority voting given a window size
    - it remove short events <= marge
    - it merge events with the same label if they are close < marge
    """

    if marge == 0 or len(pred) <= marge * 2:  # no smoothing
        return pred

    smooth_pred = [0] * marge
    for i in range(marge, len(pred) - marge):
        segment = pred[i - marge : i + marge + 1]
        smooth_pred.append(max(set(segment), key=segment.count))
    smooth_pred += [0] * marge
    assert len(smooth_pred) == len(pred), f"smooth_pred {len(smooth_pred)} != pred {len(pred)}"

    return smooth_pred


# implementation of the hungarian algorithm to find the best match
def compute_event_overlap(pred_event, gt_event):
    # compute the overlap between two events

    # the first element is the start time and the second element is the end time
    set_pred = set(range(pred_event["start"], pred_event["end"] + 1))
    set_true = set(range(gt_event["start"], gt_event["end"] + 1))
    intersection = set.intersection(set_pred, set_true)
    if len(intersection) > 0:
        p = len(intersection) / len(set_true)
        r = len(intersection) / len(set_pred)
        return 2 * p * r / (p + r)
    else:
        return 0


def compute_cost_matrix_events(pred_events, gt_events):
    # compute the cost matrix
    cost_matrix = np.zeros((len(pred_events), len(gt_events)))
    for i_pred, pred_event in enumerate(pred_events):
        for j_gt, gt_event in enumerate(gt_events):
            if pred_event["label"] == gt_event["label"]:
                cost_matrix[i_pred, j_gt] = 1 - compute_event_overlap(pred_event, gt_event)
            else:
                cost_matrix[i_pred, j_gt] = 2
    return cost_matrix


def mesures_for_event_matching(pred_events, gt_events):
    # compute the cost matrix
    cost_matrix = compute_cost_matrix_events(pred_events, gt_events)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind, cost_matrix


def transition_frame_func(y, y_events, transition_frame_marge=2):
    """
    Identify the transition frames at the start and end of the events
    We exlude them from the frame metrics
    """
    transition = np.zeros(len(y))

    for event in y_events:
        transition[
            max(event["start"] - transition_frame_marge, 0) : min(
                event["start"] + transition_frame_marge, len(y)
            )
        ] = 1
        transition[
            max(event["end"] - transition_frame_marge, 0) : min(
                event["end"] + transition_frame_marge, len(y)
            )
        ] = 1
    return transition


####
# OLD FUNCTION for event detection
####


def pdv(num, denom, defv=0):
    """
    Handls ZeroDivisionError
    """
    try:
        if num == 0.0:
            return 0.0
        return num / denom
    except ZeroDivisionError:
        print("Warning: Encountred Zero Division Error")
        return defv


def get_nb_event(event, label):
    return len([e for e in event if e["label"] == label])


def prec_recall_class(results, y_pred_event, y_true_event, key, labels_id, label_id_to_name):
    # compute precision, recall and f1 per label
    for label_id in labels_id:
        l = label_id_to_name[label_id]
        if l == "None":
            continue
        # results[l][key] = {}
        results[l][key]["nb_pred"] = get_nb_event(y_pred_event, label_id)
        results[l][key]["nb_true"] = get_nb_event(y_true_event, label_id)
        results[l][key]["precision"] = pdv(results[l][key]["p_event"], results[l][key]["nb_pred"])
        results[l][key]["recall"] = pdv(results[l][key]["r_event"], results[l][key]["nb_true"])
        p = results[l][key]["precision"]
        r = results[l][key]["recall"]
        results[l][key]["f1"] = pdv(2 * p * r, p + r)

    return results


def prec_recall_micro(results, key, label_names):
    results["micro"] = {key: {}}
    results["micro"][key]["recall"] = pdv(
        sum([results[l][key]["r_event"] for l in label_names]),
        sum([results[l][key]["nb_true"] for l in label_names]),
    )
    results["micro"][key]["precision"] = pdv(
        sum([results[l][key]["p_event"] for l in label_names]),
        sum([results[l][key]["nb_pred"] for l in label_names]),
    )
    p = results["micro"][key]["precision"]
    r = results["micro"][key]["recall"]
    results["micro"][key]["f1"] = pdv(2 * p * r, p + r)

    return results


def prec_recall_macro(results, key, label_names):
    # macro average
    results["macro"] = {key: {}}
    results["macro"][key]["recall"] = pdv(
        sum([results[l][key]["recall"] for l in label_names]), len(label_names)
    )
    results["macro"][key]["precision"] = pdv(
        sum([results[l][key]["precision"] for l in label_names]), len(label_names)
    )
    results["macro"][key]["f1"] = pdv(
        sum([results[l][key]["f1"] for l in label_names]), len(label_names)
    )

    return results


def prec_recall_weighted(results, key, label_names):
    # weighted average
    results["weighted"] = {key: {}}
    results["weighted"][key]["recall"] = pdv(
        sum([results[l][key]["recall"] * results[l][key]["nb_true"] for l in label_names]),
        sum([results[l][key]["nb_true"] for l in label_names]),
    )
    results["weighted"][key]["precision"] = pdv(
        sum([results[l][key]["precision"] * results[l][key]["nb_pred"] for l in label_names]),
        sum([results[l][key]["nb_pred"] for l in label_names]),
    )
    p = results["weighted"][key]["precision"]
    r = results["weighted"][key]["recall"]
    results["weighted"][key]["f1"] = pdv(2 * p * r, p + r)
    return results


# --------------------------------------------------------------------
def event_matching_v1(
    y_pred_event, y_true_event, label_names, labels_id, label_id_to_name, threshold=0.1, key="org"
):
    # results = {l:{'p_event': 0, 'r_event': 0} for l in label_names}
    results = {l: {key: {"p_event": 0, "r_event": 0}} for l in label_names}

    for pred_event in y_pred_event:
        # Check if the event is in the true event
        for true_event in y_true_event:
            if (pred_event["fold"] == true_event["fold"]) and (
                pred_event["label"] == true_event["label"]
            ):
                set_pred = set(range(pred_event["start"], pred_event["end"] + 1))
                set_true = set(range(true_event["start"], true_event["end"] + 1))
                intersection = set.intersection(set_pred, set_true)
                if len(intersection) > 0:
                    p = len(intersection) / len(set_true)
                    r = len(intersection) / len(set_pred)
                    f = 2 * p * r / (p + r)
                    if f >= threshold:
                        results[label_id_to_name[pred_event["label"]]][key]["p_event"] += 1
                        break

    for true_event in y_true_event:
        # Check if the event is in the true event
        for pred_event in y_pred_event:
            if (pred_event["fold"] == true_event["fold"]) and (
                pred_event["label"] == true_event["label"]
            ):
                set_pred = set(range(pred_event["start"], pred_event["end"] + 1))
                set_true = set(range(true_event["start"], true_event["end"] + 1))
                intersection = set.intersection(set_pred, set_true)
                if len(intersection) > 0:
                    p = len(intersection) / len(set_true)
                    r = len(intersection) / len(set_pred)
                    f = 2 * p * r / (p + r)
                    if f >= threshold:
                        results[label_id_to_name[pred_event["label"]]][key]["r_event"] += 1
                        break

    # class metrics
    results = prec_recall_class(
        results, y_pred_event, y_true_event, key, labels_id, label_id_to_name
    )
    # micro
    results = prec_recall_micro(results, key, label_names)
    # macro
    results = prec_recall_macro(results, key, label_names)
    # weighted
    results = prec_recall_weighted(results, key, label_names)
    # confusion
    results["confusion"] = {key: {}}

    return results



if __name__ == "__main__":
    
    # generate a random vector 
    x = torch.randn(100, 3)
    x = torch.nn.functional.normalize(x, p=2, dim=1)
    # convert to spherical coordinates
    x_spherical = cartesial2spherical(x)
    x_cart = spherical2cartesial(x_spherical)

    # check if the conversion is correct
    assert torch.allclose(x, x_cart, atol=1e-6), "Conversion failed"

    x = [[0.5,0.5,-1]]
    x = torch.nn.functional.normalize(torch.tensor(x), p=2, dim=1)
    x_spherical = cartesial2spherical(x)
    x_spherical = x_spherical*torch.tensor([-1,1])
    x_cart = spherical2cartesial(x_spherical)

    print(x, x_cart)