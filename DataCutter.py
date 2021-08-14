import jieba


class DataCutter:
    def cut(self, sen):
        dict = jieba.cut(sen)
        self.sen_cut = []
        for element in dict:
            self.sen_cut.append(element)

        return self.sen_cut

    def save(self):
        f = open('./data/data.txt', 'a')
        f.write('+++$+++ ')
        for element in self.sen_cut:
            f.write(element)
            f.write(' ')
        f.write('\n')
        f.close()

