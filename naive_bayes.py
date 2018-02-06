# PLEASE READ - You must use Python3
# Only use what is provided in the standard libraries.

import math
import os
import sys

DEFAULT_DATA_DIR = './data/'
SPAM_PATH = './train/spam/'
HAM_PATH = './train/ham/'
TEST_DATA_PATH = './test/'


def token_set(filename):
    """ This function reads in a file and returns a
        set of all the tokens. It ignores the subject line

        If the email had the following content:

        Subject: Get rid of your student loans
        Hi there,
        If you work for us, we will give you money
        to repay your student loans. You will be
        debt free!
        FakePerson_22393

        This function would return to you
        set(['', 'work', 'give', 'money', 'rid', 'your', 'there,',
            'for', 'Get', 'to', 'Hi', 'you', 'be', 'we', 'student',
            'debt', 'loans', 'loans.', 'of', 'us,', 'will', 'repay',
            'FakePerson_22393', 'free!', 'You', 'If'])
    """
    # open the file handle
    with open(filename, 'r') as f:
        text = f.read()[9:]  # Ignoring 'Subject:'
        text = text.replace('\r', '')
        text = text.replace('\n', ' ')
        tokens = text.split(' ')
        return set(tokens)


def get_prob(freq, total):
    return (freq + 1) / (total + 2)


def get_p_word_given_label(data_dir):
    """
    Returns two dictionaries each representing the
    Laplace-smoothed frequencies of words appearing
    in the sample training email data in data_dir
    """
    spam_word_freq = {}
    ham_word_freq = {}
    spam_samples = os.listdir(data_dir + SPAM_PATH)
    ham_samples = os.listdir(data_dir + HAM_PATH)
    for spam in spam_samples:
        for word in token_set(data_dir + SPAM_PATH + spam):
            if word not in spam_word_freq:
                spam_word_freq[word] = 0
            if word not in ham_word_freq:
                ham_word_freq[word] = 0
            spam_word_freq[word] = spam_word_freq[word] + 1
    for ham in ham_samples:
        for word in token_set(data_dir + HAM_PATH + ham):
            if word not in spam_word_freq:
                spam_word_freq[word] = 0
            if word not in ham_word_freq:
                ham_word_freq[word] = 0
            ham_word_freq[word] = ham_word_freq[word] + 1
    p_word_given_spam = {k: get_prob(v, len(spam_samples)) for k, v in spam_word_freq.items()}
    p_word_given_ham = {k: get_prob(v, len(ham_samples)) for k, v in ham_word_freq.items()}
    return p_word_given_spam, p_word_given_ham


def main():
    data_dir = DEFAULT_DATA_DIR
    if len(sys.argv) >= 2:
        data_dir = sys.argv[1]
        if data_dir[-1] != '/':
            data_dir = data_dir + '/'

    spam_dir = data_dir + SPAM_PATH
    ham_dir = data_dir + HAM_PATH
    test_dir = data_dir + TEST_DATA_PATH

    p_word_given_spam, p_word_given_ham = get_p_word_given_label(data_dir)

    spam_sample_count = len(os.listdir(spam_dir))
    ham_sample_count = len(os.listdir(ham_dir))
    total_sample_count = spam_sample_count + ham_sample_count
    p_spam = spam_sample_count / total_sample_count
    p_ham = ham_sample_count / total_sample_count

    test_files = os.listdir(test_dir)
    test_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # sort the files

    for file in test_files:
        log_prod_spam = 0
        log_prod_ham = 0
        tokens = token_set(test_dir + file)

        for word in tokens:
            if word in p_word_given_spam:
                log_prod_spam += math.log(p_word_given_spam[word])
                log_prod_ham += math.log(p_word_given_ham[word])

        log_spam = log_prod_spam + math.log(p_spam)
        log_ham = log_prod_ham + math.log(p_ham)

        if log_ham >= log_spam:
            print("{!s} ham".format(file))
        else:
            print("{!s} spam".format(file))


if __name__ == '__main__':
    main()
