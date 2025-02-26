parameters_filtering_bookv2 = {
    "document_length_min_cutoff": 100,
    "document_length_max_cutoff": 100000000,
    "character_repetition_max_cutoff": 0.4,
    "special_characters_max_cutoff": 0.5,
    "char_entropy_min_cutoff": 3.0,
    "lang_freq_min_cutoff": 0.4,
    "lang_score_min_cutoff": 0.6,
}

parameters_filtering = {
    "default": None,
    "book-v2": parameters_filtering_bookv2,
}
