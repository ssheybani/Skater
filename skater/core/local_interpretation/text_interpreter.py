def relevance_wt_transformer(wts_as_dict):
    # normalize score by absolute max value
    if isinstance(wts_as_dict, dict):
        N = np.max(np.abs(list(scores.values())))
        scores_dict = {word: scores[word] / N for word in scores}
        # transform dict into word list with scores
        scores = []
        for word in re.findall(r'[\w-]+', text, re.UNICODE):
            word_pp = preprocess_text(word)
            if word_pp in scores_dict:
                scores.append((word, scores_dict[word_pp]))
            else:
                scores.append((word, None))
    else:
        N = np.max(np.abs([t[1] for t in scores if t[1] is not None]))
        scores = [(w, s / N) if s is not None else (w, None) for w, s in scores]

    return scores


def show_in_notebook():
    from IPython.core.display import display, HTML
    return HTML('./rendered.html')