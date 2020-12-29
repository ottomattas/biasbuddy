#!/usr/bin/python
# -*- coding: utf-8 -*-
# Filippo Libardi & Otto Mättas
# This script will download all subreddit mentions for the given query.

# Versioning table
# v0.1 Basic elements, including local debug cache and hardcoded parameters

import sys
from psaw import PushshiftAPI

# Define the API
api = PushshiftAPI()

# Define the generator
gen = api.search_comments(q='"Alaska Native"')

# Define local cache for debugging
max_response_cache = 10
cache = []

# Print starting marker
print('Downloading...')

# Open a file to write the output to
with open('output.txt', 'w') as f:
    for c in gen:
        print(c.subreddit, file=f)
        # Uncomment this debug test to return all results
        # Wouldn't recommend it though: could take a while, but you do you
        """
        cache.append(c)
        if len(cache) >= max_response_cache:
            break
        """

    # Pick up where we left off to get the rest of the results.
    if False:
        for c in gen:
            print(c.subreddit, file=f)
            """
            cache.append(c)
            """

# Print ending marker
print('Download finished.')
