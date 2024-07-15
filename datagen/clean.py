# -=encoding=utf-8=-
import json

from re import template
from datasets import load_dataset
import json
from markdownify import markdownify
import mdformat

import re

import random
from tqdm import tqdm
from functools import lru_cache


def replace_image_tags_with_alt_text(md_text):
    # 图片标签的正则表达式
    img_pattern = r'!\[([^\]]*)\]\(([^\)]*)\)'
    
    # 使用alt_text替换
    new_md_text = re.sub(img_pattern, r'\1', md_text)
    
    return new_md_text

def replace_link_tags_with_alt_text(md_text):
    # 链接标签的正则表达式
    link_pattern = r'\[([^\]]*)\]\(([^\)]*)\)'
    
    # 使用alt_text替换
    new_md_text = re.sub(link_pattern, r'\1', md_text)
    
    return new_md_text



import re
from pylatexenc.latex2text import LatexNodes2Text

def replace_math_codes_with_text(md_text):
    # 多行数学公式标签的正则表达式
    multiline_math_pattern = r'\$\$(.*?)\$\$'
    
    # 使用LaTeX公式的普通文本替换
    new_md_text = re.sub(multiline_math_pattern, lambda m: "\n$$\n"+LatexNodes2Text().latex_to_text(m.group(1)) +"\n$$\n", md_text, flags=re.DOTALL)
    
    return new_md_text

def replace_multiline_math_tags_with_text(md_text):
    # 多行数学公式标签的正则表达式
    multiline_math_pattern = r'\$\$(.*?)\$\$'
    
    # 使用LaTeX公式的普通文本替换
    new_md_text = re.sub(multiline_math_pattern, lambda m: "\n$$\n"+LatexNodes2Text(math_mode='with-delimiters').latex_to_text(m.group(1)) +"\n$$\n", md_text, flags=re.DOTALL)
    
    return new_md_text

import re
from pylatexenc.latex2text import LatexNodes2Text

def replace_math_tags_with_text(md_text):
    # 数学公式标签的正则表达式
    math_pattern = r'\$(.*?)\$'
    
    # 使用LaTeX公式的普通文本替换
    new_md_text = re.sub(math_pattern, lambda m: "$"+LatexNodes2Text(math_mode='with-delimiters').latex_to_text(m.group(1)) +"$", md_text)
    
    return new_md_text

import re

def remove_http_links(text):
    # HTTP链接的正则表达式
    http_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # 使用空字符串替换所有HTTP链接
    cleaned_text = re.sub(http_pattern, '', text)
    
    return cleaned_text

import html
import re

def html_to_markdown(html_text):
    # Unescape HTML special characters
    unescaped_text = html.unescape(html_text)

    # Replace HTML tags with Markdown equivalent
    markdown_text = re.sub(r'<b>(.*?)</b>', r'**\1**', unescaped_text)  # Bold
    markdown_text = re.sub(r'<i>(.*?)</i>', r'*\1*', markdown_text)  # Italic
    markdown_text = re.sub(r'<sub>(.*?)</sub>', r'~\1~', markdown_text)  # Subscript
    markdown_text = re.sub(r'<sup>(.*?)</sup>', r'^\1^', markdown_text)  # Superscript
    markdown_text = re.sub(r'<del>(.*?)</del>', r'~~\1~~', markdown_text)  # Strikethrough
    markdown_text = re.sub(r'<a href="(.*?)">(.*?)</a>', r'[\2](\1)', markdown_text)  # Link

    return markdown_text

# def clean(match):
#     # 在这里添加你的清洗逻辑
#     smiles_string = match.group(0)
#     if '.' in smiles_string:
#         return smiles_string
#     else:
#         cleaned_smiles = f'\n```smiles\n{smiles_string}\n```\n'
#         return cleaned_smiles

# def replace_smiles_with_cleaned(text):
#     # SMILES字符串的正则表达式
#     smiles_pattern = r'[A-Za-z0-9@+\-\[\]\(\)=%#:.]*'

#     # 使用re.sub()替换所有的SMILES字符串
#     cleaned_text = re.sub(smiles_pattern, clean, text)

#     return cleaned_text

# def md(txt):
#     txt = html_to_markdown(txt)
#     txt = remove_http_links(txt)
#     txt = replace_multiline_math_tags_with_text(txt)
#     txt = replace_math_tags_with_text(txt)
#     # txt = LatexNodes2Text(math_mode='with-delimiters').latex_to_text(txt)
#     txt = txt.replace('\\[','[').replace('\\]',']').replace('\\_','_')

#     txt = markdownify(txt)
#     # txt = mdformat.text(txt)
    
#     txt = replace_image_tags_with_alt_text(txt)
#     txt = replace_link_tags_with_alt_text(txt)
#     # txt = replace_link_tags_with_alt_text(txt)
#     txt = mdformat.text(txt)
#     txt = txt.replace('lt;','<').replace('gt;','>').replace('amp;','&').replace('quot;','"').replace('apos;',"'").replace('nbsp;',' ').replace('ldquo;','“').replace('rdquo;','”').replace('lsquo;','‘').replace('rsquo;','’').replace('mdash;','—').replace('ndash;','–').replace('times;','×').replace('divide;','÷').replace('leq;','≤').replace('geq;','≥').replace('neq;','≠').replace('infty;','∞').replace('alpha;','α').replace('beta;','β').replace('gamma;','γ').replace('delta;','δ').replace('epsilon;','ε').replace('zeta;','ζ').replace('eta;','η').replace('theta;','θ').replace('iota;','ι').replace('kappa;','κ').replace('lambda;','λ').replace('mu;','μ').replace('nu;','ν').replace('xi;','ξ').replace('omicron;','ο').replace('pi;','π').replace('rho;','ρ').replace('sigma;','σ').replace('tau;','τ').replace('upsilon;','υ').replace('phi;','φ').replace('chi;','χ').replace('psi;','ψ').replace('omega;','ω').replace('Alpha;','Α').replace('Beta;','Β').replace('Gamma;','Γ').replace('Delta;','Δ').replace('Epsilon;','Ε').replace('Zeta;','Ζ').replace('Eta;','Η').replace('Theta;','Θ').replace('Iota;','Ι').replace('Kappa;','Κ').replace('Lambda;','Λ').replace('Mu;','Μ').replace('Nu;','Ν').replace('Xi;','Ξ').replace('Omicron;','Ο').replace('Pi;','Π').replace('Rho;','Ρ').replace('Sigma;','Σ').replace('Tau;','Τ').replace('Upsilon;','Υ').replace('Phi;','Φ').replace('Chi;','Χ').replace('Psi;','Ψ').replace('Omega;','Ω').replace('forall;','∀').replace('part;','∂').replace('exist;','∃').replace('empty;','∅').replace('nabla;','∇').replace('isin;','∈').replace('notin;','∉').replace('ni;','∋').replace('prod;','∏').replace('sum;','∑').replace('minus;','−').replace('lowast;','∗').replace('radic;','√').replace('prop;','∝').replace('infin;','∞').replace('ang;','∠').replace('and;','∧').replace('or;','∨').replace('cap;','∩').replace('cup;','∪').replace('int;','∫').replace('there4;','∴').replace('sim;','∼').replace('cong;','≅').replace('asymp;','≈').replace('ne;','≠').replace('equiv;','≡').replace('le;','≤').replace('ge;','≥').replace('sub;','⊂').replace('sup;','⊃').replace('nsub;','⊄').replace('sube;','⊆').replace('supe;','⊇').replace('oplus;','⊕').replace('otimes;','⊗').replace('perp;','⊥').replace('sdot;','⋅').replace('lceil;','⌈').replace('rceil;','⌉').replace('lfloor;','⌊').replace('rfloor;','⌋').replace('lang;','⟨').replace('rang;','⟩').replace('loz;','◊').replace('spades;','♠').replace('clubs;','♣').replace('hearts;','♥').replace('diams;','♦').replace('quot;','"').replace('amp;','&').replace('lt;','<').replace('gt;','>').replace('nbsp;',' ').replace('iexcl;','¡').replace('cent;','¢').replace('pound;','£').replace('curren;','¤')
#     txt = txt.replace('yen;','¥').replace('brvbar;','¦').replace('sect;','§').replace('uml;','¨')
#     # if 'smiles' in txt or 'SMILES' in txt:
#     #     txt = replace_smiles_with_cleaned(txt)
#     # txt = txt.split('References:')[0]
#     txt = txt.replace('\\[','[').replace('\\]',']').replace('\\_','_')
#     txt = txt.replace('enter image description here','')
#     txt = txt.replace('$$$$','')
#     return txt

import re

def replace_subscripts(text):
    # The regular expression pattern to find all $_{2}$, $_{3}$, etc.
    pattern = r'\$_{(.*?)}\$\s*'
    
    # The replacement pattern _{2}, _{3}, etc.
    replacement = r'_{\1}'
    
    # Use re.sub() to find and replace all matches
    return re.sub(pattern, replacement, text)

def replace_chinese_punctuation(text):
    # text = text.replace('，', ',')
    # text = text.replace('。', '.')
    # text = text.replace('！', '!')
    # text = text.replace('？', '?')
    # text = text.replace('：', ':')
    # text = text.replace('；', ';')
    # text = text.replace('“', '"')
    # text = text.replace('”', '"')
    # text = text.replace('‘', "'")
    # text = text.replace('’', "'")
    text = text.replace('（', '(')
    text = text.replace('）', ')')
    # text = text.replace('【', '[')
    # text = text.replace('】', ']')
    # text = text.replace('《', '<')
    # text = text.replace('》', '>')
    # text = text.replace('——', '--')
    # text = text.replace('、',',')
    # text = text.replace('．','.')
    # text = text.replace('..','.')
    return text

import re

def replace_text_nested_braces(text):
    # The regular expression pattern to find all nested braces with \\rm A
    pattern = r'\{+\\\\rm\s(.*?)\}+'
    
    # The replacement pattern $A$
    replacement = r'$\1$'
    
    # Use re.sub() to find and replace all matches
    return re.sub(pattern, replacement, text)

def md(txt):
    txt = txt.replace('<sup>','^{').replace('</sup>','}').replace('<sub>','_{').replace('</sub>','}')
    txt = replace_chinese_punctuation(txt)
    txt = replace_subscripts(txt)
    # txt = remove_http_links(txt)

    txt = LatexNodes2Text(math_mode='with-delimiters').latex_to_text(txt)
    txt = txt.replace(' ^','^').replace(' _','_')
    # txt = txt.replace('\\[','[').replace('\\]',']').replace('\\_','_')

    # txt = markdownify(txt)
    # txt = replace_multiline_math_tags_with_text(txt)
    # txt = replace_math_tags_with_text(txt)
    # # # txt = mdformat.text(txt)
    
    # txt = replace_image_tags_with_alt_text(txt)
    # txt = replace_link_tags_with_alt_text(txt)
    # txt = replace_link_tags_with_alt_text(txt)
    # txt = mdformat.text(txt)
    txt = txt.replace('\_','_').replace('\n\n','\n')
    txt = txt.replace('$$\n','$').replace('\n$$','$').replace('$$','$').replace('$=$','=').replace('$>$','>').replace('$<$','<')
    if txt.endswith('\n'):
        txt = txt[:-1]
    # txt = txt.replace('lt;','<').replace('gt;','>').replace('amp;','&').replace('quot;','"').replace('apos;',"'").replace('nbsp;',' ').replace('ldquo;','“').replace('rdquo;','”').replace('lsquo;','‘').replace('rsquo;','’').replace('mdash;','—').replace('ndash;','–').replace('times;','×').replace('divide;','÷').replace('leq;','≤').replace('geq;','≥').replace('neq;','≠').replace('infty;','∞').replace('alpha;','α').replace('beta;','β').replace('gamma;','γ').replace('delta;','δ').replace('epsilon;','ε').replace('zeta;','ζ').replace('eta;','η').replace('theta;','θ').replace('iota;','ι').replace('kappa;','κ').replace('lambda;','λ').replace('mu;','μ').replace('nu;','ν').replace('xi;','ξ').replace('omicron;','ο').replace('pi;','π').replace('rho;','ρ').replace('sigma;','σ').replace('tau;','τ').replace('upsilon;','υ').replace('phi;','φ').replace('chi;','χ').replace('psi;','ψ').replace('omega;','ω').replace('Alpha;','Α').replace('Beta;','Β').replace('Gamma;','Γ').replace('Delta;','Δ').replace('Epsilon;','Ε').replace('Zeta;','Ζ').replace('Eta;','Η').replace('Theta;','Θ').replace('Iota;','Ι').replace('Kappa;','Κ').replace('Lambda;','Λ').replace('Mu;','Μ').replace('Nu;','Ν').replace('Xi;','Ξ').replace('Omicron;','Ο').replace('Pi;','Π').replace('Rho;','Ρ').replace('Sigma;','Σ').replace('Tau;','Τ').replace('Upsilon;','Υ').replace('Phi;','Φ').replace('Chi;','Χ').replace('Psi;','Ψ').replace('Omega;','Ω').replace('forall;','∀').replace('part;','∂').replace('exist;','∃').replace('empty;','∅').replace('nabla;','∇').replace('isin;','∈').replace('notin;','∉').replace('ni;','∋').replace('prod;','∏').replace('sum;','∑').replace('minus;','−').replace('lowast;','∗').replace('radic;','√').replace('prop;','∝').replace('infin;','∞').replace('ang;','∠').replace('and;','∧').replace('or;','∨').replace('cap;','∩').replace('cup;','∪').replace('int;','∫').replace('there4;','∴').replace('sim;','∼').replace('cong;','≅').replace('asymp;','≈').replace('ne;','≠').replace('equiv;','≡').replace('le;','≤').replace('ge;','≥').replace('sub;','⊂').replace('sup;','⊃').replace('nsub;','⊄').replace('sube;','⊆').replace('supe;','⊇').replace('oplus;','⊕').replace('otimes;','⊗').replace('perp;','⊥').replace('sdot;','⋅').replace('lceil;','⌈').replace('rceil;','⌉').replace('lfloor;','⌊').replace('rfloor;','⌋').replace('lang;','⟨').replace('rang;','⟩').replace('loz;','◊').replace('spades;','♠').replace('clubs;','♣').replace('hearts;','♥').replace('diams;','♦').replace('quot;','"').replace('amp;','&').replace('lt;','<').replace('gt;','>').replace('nbsp;',' ').replace('iexcl;','¡').replace('cent;','¢').replace('pound;','£').replace('curren;','¤')
    # txt = txt.replace('yen;','¥').replace('brvbar;','¦').replace('sect;','§').replace('uml;','¨')
    # # if 'smiles' in txt or 'SMILES' in txt:
    # #     txt = replace_smiles_with_cleaned(txt)
    # # txt = txt.split('References:')[0]
    # txt = txt.replace('\\[','[').replace('\\]',']').replace('\\_','_')
    # txt = txt.replace('enter image description here','')
    # txt = txt.replace('$$$$','')
    return txt

# data = []
# with open("./general_dpo_data_cn.jsonl", 'r',encoding='utf8') as f:
#     with open("./cleaned_general_dpo_data_cn.jsonl", 'w+',encoding='utf8') as f0:
#         for line in f:
#             # if any([ i in line.lower() for i in ['编写','代码','翻译',"脚本",'编程','程序','print','if','IF','python']]):
#             #     continue
#             # else:
#             data = json.loads(line)
#             if data['instruction'] != '':
#                 data['instruction'] = md(data['instruction'])
#             if data['input'] != '':
#                 data['input'] = md(data['input'])
#             if data['output'] != []:
#                 data['output'] = [md(data['output'][0]),md(data['output'][1])]
#             if data['history']:
#                 history = []
#                 for qa in data['history']:
#                     history.append([md(qa[0]),md(qa[1])])
#                 data['history'] = history
#             f0.write(json.dumps(data,ensure_ascii=False)+'\n')

# data = []
# with open("./text_pure.jsonl", 'r',encoding='utf8') as f:
#     with open("./cleaned_text_pure.jsonl", 'w+',encoding='utf8') as f0:
#         for line in tqdm(f):
#             # if any([ i in line.lower() for i in ['编写','代码','翻译',"脚本",'编程','程序','print','if','IF','python']]):
#             #     continue
#             # else:
#             data = json.loads(line)
#             messages = data['conversations']
#             for mes in messages:
#                 mes['value'] = md(mes['value'])
#             data['conversations'] = messages
#             f0.write(json.dumps(data,ensure_ascii=False)+'\n')

# print(md('3NH<sub>3</sub>'))
# print(md("{\\rm CN ^{-}}"))
# print(md('标准答案为: A. {\\rm A}1. ${{\\rm SO _{3} ^{2-}}}$原子数目为{\\rm 4}，价电子数目{\\rm 6+6×3+2=26}，${{\\rm PO _{3} ^{3-}}}$原子数目为{\\rm 4}，价电子数目{\\rm 5+6×3+3=26}，二者为等电子体； 2. ${{\\rm NO ^{+}}}$原子数目为{\\rm 2}，价电子数目{\\rm 5+6-1=10}，${{\\rm CN ^{-}}}$原子数目为{\\rm 2}，价电子数目{\\rm 4+5+1=10}，二者为等电子体； 3. ${{\\rm CO _{2}}}$原子数目为{\\rm 3}，价电子数目{\\rm 4+6×2=16}，${{\\rm CS _{2}}}$原子数目为{\\rm 3}，价电子数目{\\rm 4+6×2=16}，二者为等电子体； 4. ${{\\rm NO _{2}}}$原子数目为{\\rm 3}，价电子数目{\\rm 5+6×2=17}，${{\\rm CO _{2}}}$原子数目为{\\rm 3}，价电子数目{\\rm 4+6×2=16}，二者不是等电子体，则只有1. 2. 3. 是等电子体， 故选：{\\rm A}。具有相同价电子数和原子数目的微粒为等电子体，以此来解答。本题考查等电子体，为高频考点，把握微粒中的原子数、价电子数为解答的关键，侧重分析与应用能力的考查，注意价电子判断，题目难度不大。'))
# print(md('标准答案为: 二氧化碳、CO2+H2O=H2CO3．. （1）根据课本知识可知，氢气、甲烷、一氧化碳具有可燃性；（2）能用向上排空气法收集的气体是密度大于空气密度的气体；（3）在五中气体中只有二氧化碳使澄清石灰水变浑浊，然后根据反应物和生成物以及反应条件写出反应式；（4）使紫色石蕊试液变红的是酸性物质，以上气体中只有二氧化碳可使水溶液显酸性．（1）根据课本知识可知，氢气、甲烷、一氧化碳具有可燃性；纯净的氢气在空气中安静的燃烧，发出淡蓝色的火焰；甲烷在空气中燃烧，产生淡绿色的火焰，放出大量的热；一氧化碳在空气中燃烧，产生淡蓝色火焰，放出热量．故为：氢气、甲烷、一氧化碳；（2）能用向上排空气法收集的气体是密度大于空气密度的气体．根据课本所学知识可知，氧气和二氧化碳的密度大于空气的密度，氢气、甲烷和一氧化碳的密度小于空气的密度．所以能与向上排空气法收集的气体是氧气和二氧化碳．故为：氧气、二氧化碳；（3）在五种气体中只有二氧化碳使澄清石灰水变浑浊，这是二氧化碳的特性．石灰水的主要成分是Ca（OH）$_{2}$，与二氧化碳反应生成碳酸钙沉淀和水，反应式为：CO$_{2}$+Ca（OH）$_{2}$=CaCO$_{3}$↓+H$_{2}$O．故为：二氧化碳、CO$_{2}$+Ca（OH）$_{2}$=CaCO$_{3}$↓+H$_{2}$O；（4）能使紫色石蕊试液变红的是显酸性的物质．以上五种气体中能溶于水且溶液显酸性的只有二氧化碳，它溶于水后可生成碳酸，溶液成酸性．反应式为：CO$_{2}$+H$_{2}$O=H$_{2}$CO$_{3}$．故为：二氧化碳、CO$_{2}$+H$_{2}$O=H$_{2}$CO$_{3}$．'))