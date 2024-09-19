def remove_not_selected(item: list) -> list:
    """
    Helper function for the ChiSquaredTester class. It removes the string "not selected" from any list.

    Parameters
    ----------
    item : list
        Any list.

    Returns
    -------
    item : list
        The same list but items of "not selected" were removed if present.
    """
    try:
        while True:
            item.remove("not selected")
    except ValueError:
        pass

    return item


def remove_not_select_and_cat_percentage(row):
    element_to_remove = "not selected"
    if element_to_remove in row['sig_more_category']:
        index_to_remove = row['sig_more_category'].index(element_to_remove)
        row['sig_more_category'].remove(element_to_remove)
        row['category_percentages'].pop(index_to_remove)
    return row
