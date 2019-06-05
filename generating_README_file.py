import os

def read_README_file():
    with open("README.md", 'r', encoding='utf-8') as f:
        readme_file = f.readlines()
        print(readme_file)

def readme_file_head():
    content = ["# Content\n", "机器学习深度学习相关书籍、课件、代码的仓库。\n", "Machine learning is the warehouse of books, courseware and codes.\n"]
    return content

def get_PDF_file_name_list(file_dir="book"):
    pdf_file_name_list =[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.pdf':
                pdf_file_name_list.append(os.path.join(root, file))
    pdf_file_name_list = [str(idx+1) + ". " + name + "\n" for idx, name in enumerate(pdf_file_name_list)]
    pdf_file_name_list.insert(0, "## book\n")
    return pdf_file_name_list

def get_Courseware_file_name_list(file_dir="Courseware"):
    Courseware_file_name_list =[]
    for root, dirs, files in os.walk(file_dir):
        Courseware_file_name_list.extend(dirs)
    Courseware_file_name_list = [str(idx+1) + ". " + name + "\n" for idx, name in enumerate(Courseware_file_name_list)]
    Courseware_file_name_list.insert(0, "## Courseware\n")
    return Courseware_file_name_list


def wirte_README_file():
    readme_file_head_list = readme_file_head()
    pdf_file_name_list = get_PDF_file_name_list()
    Courseware_file_name_list = get_Courseware_file_name_list()
    content_list = [readme_file_head_list, pdf_file_name_list, Courseware_file_name_list]
    with open("README.md", 'w', encoding='utf-8') as wf:
        for content in content_list:
            wf.writelines(content)
            wf.write("\n")

if __name__=="__main__":
    wirte_README_file()