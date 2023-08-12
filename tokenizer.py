

def basic_tokenizer(vocabs):
    vocab_to_id = {v:i+259 for i, v in enumerate(vocabs)}
    id_to_vocab = {v:k for k, v in vocab_to_id.items()}
    def encode(text):
        token_list = []
        for token in text:
            if token in vocab_to_id:
                token_list.append(vocab_to_id[token])
            else:
                for c in token.encode('utf-16 be'):
                    token_list.append(c+3)
        return token_list
    def decode(ids):
        text = ''
        i = 0
        while i < len(ids):
            if ids[i] >= 259:
                text += id_to_vocab[ids[i]]
                i+=1
            elif ids[i] > 2:
                char = (ids[i]-3).to_bytes(1, 'big')
                i+=1
                while i < len(ids) and ids[i] > 2 and ids[i] < 259:
                    char += (ids[i]-3).to_bytes(1, 'big')
                    i+=1
                try:
                    text += char.decode('utf-16 be')
                except:
                    continue
            else:
                i+=1
        return text
    return encode, decode
