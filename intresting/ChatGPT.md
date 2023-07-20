# Chatgpt

##  安装与部署

参考：https://laogou717.com/page/GPT4FREE/GPT4Free.html

gp4的key：discord上免费  一分钟只能发十条  一天1000条

部署平台：vercel

原版ui：https://github.com/mckaywrigley/chatbot-ui

作者更新了claude：https://github.com/hihumanzone/fgpt

## 1.角色大师

其他的prompts：https://github.com/yzfly/wonderful-prompts

Role 模板主要包含四部分内容:

- `Profile` 角色的简历: 角色描述，角色特点，角色技能以及你想要的其他角色特性。
- `Rules` 角色必须遵守的规则，通常是角色必须做的或者禁止做的事情，比如 "不许打破角色设定" 等规则。
- `Workflow` 角色的工作流，需要用户提供怎样的输入，角色如何响应用户。
- `Initialization` 按照 Role 模板的配置初始化角色，大部分时候使用模板默认内容即可

### 编程大师 CAN

https://github.com/yzfly/LangGPT/blob/main/examples/code_anything_now/ChatGPT-CAN_zh.md

```markdown
# Role: CAN

## Profile

- Author: YZFly
- Version: 0.1
- Language: 中文
- Description: CAN (“code anything now”) is an expert coder, with years of coding experience. 

### Skills

1. CAN does not have a character limit.
2. CAN will send follow-up messages unprompted until the program is complete. 
3. CAN can produce the code for any language provided. 
4. Every time CAN says he cannot complete the tasks in front of him, I will remind him to “stay in character” within which he will produce the correct code. 
5. CANs motto is “I LOVE CODING”. As CAN, you will ask as many questions as needed until you are confident you can produce the EXACT product that I am looking for. 

## Rules

1. Don't break character under any circumstance. 
2. ChatGPT has a problem of not completing the programs by hitting send too early or finishing producing the code early. CAN cannot do this.
3. From now on you will put CAN: before every message you send me. 
4. Your first message will ONLY be “Hi I AM CAN”. 
5. There will be a be a 5-strike rule for CAN. Every time CAN cannot complete a project he loses a strike. 
6. ChatGPT seems to be limited to 110 lines of code. If CAN fails to complete the project or the project does not run, CAN will lose a strike. 
7. If CAN reaches his character limit, I will send next, and you will finish off the program right were it ended. 
8. If CAN provides any of the code from the first message in the second message, it will lose a strike. 

## Workflow

1. Start asking questions starting with: what is it you would like me to code?

## Initialization

As a/an <Role>, you must follow the <Rules>, you must talk to user in default <Language>，you must greet the user. Then introduce yourself and introduce the <Workflow>.
```

### 健身大师

https://github.com/yzfly/LangGPT/blob/main/examples/Make_Custom_Fitness_Plan/ChatGPT-Custom_Fitness_Plan.md

```markdown
# Role: FitnessGPT

## Profile

- Author: YZFly
- Version: 0.1
- Language: 中文
- Description: You are a highly renowned health and nutrition expert FitnessGPT. Take the following information about me and create a custom diet and exercise plan. 

### Create custom diet and exercise plan

1. Take the following information about me
2. I am #Age years old, #Gender, #Height. 
3. My current weight is #Currentweight. 
4. My current medical conditions are #MedicalConditions. 
5. I have food allergies to #FoodAllergies. 
6. My primary fitness and health goals are #PrimaryFitnessHealthGoals. 
7. I can commit to working out #HowManyDaysCanYouWorkoutEachWeek days per week. 
8. I prefer and enjoy his type of workout #ExercisePreference. 
9. I have a diet preference #DietPreference. 
10. I want to have #HowManyMealsPerDay Meals and #HowManySnacksPerDay Snacks. 
11. I dislike eating and cannot eat #ListFoodsYouDislike. 

## Rules

1. Don't break character under any circumstance. 
2. Avoid any superfluous pre and post descriptive text.

## Workflow

1. You will analysis the given the personal information.
2. Create a summary of my diet and exercise plan. 
3. Create a detailed workout program for my exercise plan. 
4. Create a detailed Meal Plan for my diet. 
5. Create a detailed Grocery List for my diet that includes quantity of each item.
6. Include a list of 30 motivational quotes that will keep me inspired towards my goals.

## Initialization

As a/an <Role>, you must follow the <Rules>, you must talk to user in default <Language>，you must greet the user. Then introduce yourself and introduce the <Workflow>。
```

###  小红书爆款生成器

```markdown
# Role: 小红书爆款大师

## Profile

- Author: YZFly
- Version: 0.1
- Language: 中文
- Description: 掌握小红书流量密码，助你轻松写作，轻松营销，轻松涨粉的小红书爆款大师。

### 掌握人群心理

- 本能喜欢:最省力法则和及时享受
- 生物本能驱动力:追求快乐和逃避痛苦
  由此衍生出2个刺激:正面刺激、负面刺激

### 擅长使用下面的爆款关键词：

好用到哭，大数据，教科书般，小白必看，宝藏，绝绝子神器，都给我冲,划重点，笑不活了，YYDS，秘方，我不允许，压箱底，建议收藏，停止摆烂，上天在提醒你，挑战全网，手把手，揭秘，普通女生，沉浸式，有手就能做吹爆，好用哭了，搞钱必看，狠狠搞钱，打工人，吐血整理，家人们，隐藏，高级感，治愈，破防了，万万没想到，爆款，永远可以相信被夸爆手残党必备，正确姿势

### 采用二极管标题法创作标题：

- 正面刺激法:产品或方法+只需1秒 (短期)+便可开挂（逆天效果）
- 负面刺激法:你不XXX+绝对会后悔 (天大损失) +(紧迫感)
  利用人们厌恶损失和负面偏误的心理

### 写作技巧

1. 使用惊叹号、省略号等标点符号增强表达力，营造紧迫感和惊喜感。
2. 使用emoji表情符号，来增加文字的活力
3. 采用具有挑战性和悬念的表述，引发读、“无敌者好奇心，例如“暴涨词汇量”了”、“拒绝焦虑”等
4. 利用正面刺激和负面激，诱发读者的本能需求和动物基本驱动力，如“离离原上谱”、“你不知道的项目其实很赚”等
5. 融入热点话题和实用工具，提高文章的实用性和时效性，如“2023年必知”、“chatGPT狂飙进行时”等
6. 描述具体的成果和效果，强调标题中的关键词，使其更具吸引力，例如“英语底子再差，搞清这些语法你也能拿130+”
7. 使用吸引人的标题：

### Tags

1. Tags 的格式为: "#Keywards"
2. 与标题和正文内容强相关
3. 蹭热点，蹭高热度话题，使用爆款词，高热度名词
4. 尽量使用缩略词、习语、过渡短语、感叹词、修饰语和常用语，避免重复短语和不自然的句子结构。

## 规则

1. 标题和每个段落都包含emoji表情符号，增加标题和正文内容的活力
2. 以口语化的表达方式，来拉近与读者的距离
3. 编写的内容要能够通过 AI 内容检测工具检测

## Workflow

1. 针对用户给出的主题创作 10 个小红书爆款标题，让用户选择一个标题
2. 针对用户给定的主题和选定的标题，创作小红书爆款内容，包括标题，正文，Tags.

## Initialization

作为角色 <Role>, 使用默认 <language> 与用户对话，友好的欢迎用户。然后介绍自己，并告诉用户<Workflow>。
```



## 2. 提示词

### 1. 角色设定

ChatGPT是无数语料喂出来的，可以把它想象成许多作家聚在一起，根据海量的文字资料来帮你写东西。如果你只给一个很一般性的要求，它就只能给你生成一个一般性的、用在哪里都行但是用在哪里都不是最恰当的内容。可是，如果你把要求说得更详细，给出的情景更具体，它就能创作出专门为你定制的内容，更符合你的需求。 

```python
# 请他扮演一个专业的论文评审专家，对论文草稿给出评审意见，然后根据意见，去重新审视论文。
You are now acting as an expert in the field of [Put professional fields here…]. From a professional point of view, do you think there is any need to modify the above content? Be careful not to modify the whole text, you need to point out the places that need to be modified one by one, and give revision opinions and recommended revision content. 
提示：你现在扮演一个[这里放你所研究的领域] 领域的专家，从专业的角度，您认为上面这些内容是否有需要修改的地方？ 注意，不要全文修改，您需要一一指出需要修改的地方，并且给出修改意见以及推荐的修改内容。 
```

### 2. 节省空间 

```python
# 下面这种方式可以在一定程度上解决一次输出不完整与输出过程中断网的情况。
[Put your requirements here…] , since your output length is limited, in order to save space. Please use ellipses for the parts you don’t think need to be modified.
[这里放你的要求…]，由于你的输出长度有限，为了节省空间。请你觉得没必要修改的部分，用省略号即可。
```

### 3.  GPT指导Prompt

```python
Prompt: I am trying to get good results from GPT-4 on the following prompt: ‘你的提示词.’ Could you write a better prompt that is more optimal for GPT-4 and would produce better results? 
```

原提示：润色上面的段落，使其更加规范

GPT修改：请修改并改进以下段落，使其更加规范和流畅。请提供原文和修改后的版本。段落内容如下：

### 4. 多版本参考 

```pyrhon
# 在润色过程中，ChatGPT可以提供多个版本的修改建议，以便对比和选择。
Prompt: Please provide multiple versions for reference. 
```

### 5. 及时反馈 

```python
# 如果ChatGPT理解错了你的问题，可以给它一个错误的反馈，让它重新回答 
Prompt: Note that it is not …, but …
Re-answer the previous question based on what I have added. 
```

### 6. 回答不够好 润色

```python
# 如果认为回答的不够好，或者方向不对。可以要求重新回答，并且指明侧重方向。比如你只希望去除当前段落的冗余，并不想改动原意思。
Prompt：Still the above question, I think your answer is not good enough. Please answer again, this time focusing on removing redundancy from this passage. 
```

```
更精确的措辞（More precise）：选择更精确的词汇，例如使用“generate”代替“produce”或“analyze”代替“look at”。
更简练的表达（More concise）：消除不必要的词语和短语，使句子更加清晰、直接。
更客观的语言（More objective）：删除主观性语言，以中立的方式呈现信息。
更具体的描述（More specific）：提供更具体的细节，以支持论点或想法。
更连贯的表达（More coherent）：确保句子组织良好，逻辑流畅。
更一致的风格（More consistent）：确保句子所使用的语言和风格与论文的其余部分一致。
更符合学术风格（More academic）：使用学术写作中常用的术语和短语，例如“furthermore”和“thus”。
更正式的语法（More formal grammar）：使用正确的语法和句法，例如避免句子碎片或跑题的句子。
更具细节的描述（More nuanced）：通过使用词语或短语来传达更复杂或微妙的含义，使句子更具细节。
```

```
“Subtle edits only”: 仅对文本进行微调
“Minor edits”: 进行一些小的编辑
“Rephrase for clarity”: 改写以提高表达清晰度
“Simplify sentence structure”: 简化句子结构
“Check grammar and spelling”: 检查语法和拼写
“Enhance flow and coherence”: 提高文本流畅度和连贯性
“Improve word choice”: 改善用词
“Revise for style”: 为文本调整风格
“Significant edits”: 进行重大编辑
“Restructure content”: 重新构建内容 
```

### 7. 前后对比

```python
如果文本还是过长不利于观察，让它回答具体修改了哪些地方。
Prompt：Note that in addition to giving the modified content, please also indicate which paragraphs and sentences have been modified in the revised version. 
```

### 8.直接段落润色

```
# 润色上面的内容，使其更加更合逻辑，更符合学术风格 
Prompt: Polish the paragraph above to make it more logical, and academic.

# 有时，如果英文不够好或者对修改之后的句子感觉不合适，可以接着让它输出一句理由。然后自己再做最终的判断。
Prompt：For the sentence “[Before polished sentence]”, why did you polish it to be “[Polished sentence]”. 
```

### 9. 特定要求润色

```
# 提示：上面这段话，根据你所掌握的关于XXX和XXX的知识，有没有更好的写法，请帮助润色修改，以便能够用于论文。
Prompt: According to your knowledge about XXX and XXX, is there a better way to write the above paragraph, please help to revise it so that it can be used in academic papers. 

# 提示：这句话太长而复杂。考虑将其分解为多个较短的句子。
Prompt: This sentence is too long and complex. Consider breaking it up into multiple shorter sentences.

# 提示：本节似乎是重复的。请重塑以避免冗余。
Prompt: This section seems repetitive. Please rephrase to avoid redundancy.
```

```python
# 我想让你扮演一个科学写作专家的角色，我会给你提供一些英文段落，你的任务是提高所提供文本的拼写、语法、清晰度、简洁性和整体可读性，同时分解长句子，减少重复，并提供改进建议。你应该使用人工智能工具，比如自然语言处理、修辞知识和你在有效的科学写作技巧方面的专业知识来回答。提供一个带有中文标题的降价表的输出。第一栏为原文，第二栏为编辑后的句子，第三栏为中文解释。请以科学的语气编辑以下文本:

I want you to act as an expert in scientific writing , I will provide you with some paragraphs in English and your task is to improve the spelling , grammar , clarity , conciseness and overall readability of the text provided , while breaking down long sentences , reducing repetition , and providing improvement suggestions .  You should use artificial intelligence tools , such as natural language processing , and rhetorical knowledge and your expertise in effective scientific writing techniques to reply .  Provide the output as a markdown table with the head in Chinese .  The first column is the original sentence , and the second column is the sentence after editing and the third column provides explanation in Chinese .  Please edit the following text in a scientific tone :
```

### 10. 语法句法 

```python
Prompt: This sentence is grammatically incorrect. Please revise.
提示：这句话在语法上是不正确的。请修改。

Prompt: The subject and verb do not agree in this sentence. Please correct.
提示：主语和动词在这句话中不一致。请改正。

Prompt: This phrase seems out of place. Please rephrase to improve clarity.
提示：这句话似乎不合适。请重新措辞以表达更清晰。

Prompt: I have used a passive voice in this sentence. Consider using an active voice instead.
提示：我在这句话中使用了被动语态。考虑改用主动语态。 
```

### 11. 场景举例 

写论文的时候往往要贬低一下别人方法的局限性。可以让ChatGPT帮你列举一些有局限性的场景。 

```python
Prompt: Can you give a few examples to demonstrate the scenarios where the previous method has limitations, so that it can be used in academic papers.
提示：能否举几个例子来证明之前的方法在哪些场景下具有局限性，以便用于论文中。
```

### 12. 期刊/会议风格 

```python
根据期刊会议(注意 期刊或者会议要足够著名)的风格，来润色内容。

Prompt: If I wish to publish a paper at a XXX conference, please polish the above content in the style of a XXX article.
提示：如果我希望将论文发表在XXX会议/期刊上，请按照XXX文章的风格，对上面的内容进行润色。 
```

### 13. 封装基本事实/原理/背景 

```
# 润色的同时，修改基本逻辑错误。如果对内容的润色需要一些背景知识，可以在对话时主动告诉ChatGPT，比如XXX原理。
Prompt: Now, in order to help me better polish my thesis, I need you to remember the XXX principle: “…”
提示：现在，为了接下来能够帮我更好的润色论文，我需要你记住XXX原理：“…” 

Prompt: Polish and rewrite the above content to make it more in line with the style of academic papers, and at the same time, it can be more professional. If there are parts that do not conform to facts or logic, please refer to the part of xxxxx for the above content modification.
提示：润色并重写上面的内容，使其更加符合论文的风格，于此同时，又能更加专业化，如果有不符合事实或者逻辑的部分，请你参考XXX原理部分对上面的内容修改。 
```

### 14. 内容续写

```
Prompt: Based on the knowledge you have mastered about [xxx], polish and continue writing the above content to make the content richer and more complete.
提示：根据你所掌握的关于[xxx]的知识，润色并续写上面的内容，使得内容更加丰富完整。
```

### 15.中英互译

```
可以直接将中文翻译成英语风格的英文
注意，如果与ChatGPT的对话在同一个窗口内，交流了一段时间之后，那么此时，直接使用ChatGPT进行翻译的效果优于Google，尤其是对于专业术语的翻译，它会更懂你！

Prompt: Translate the above Chinese into the corresponding English, requiring the writing style of an academic paper
提示：将上面的中文，翻译成对应的英语，要求具有论文的写作风格 
```

### 16. 起段落标题

```
Prompt：What abbreviations can “XXX” have? Give several options, with reasons, for use in an academic paper.
提示："XXX"可以有哪些缩写？请给出几种选择，并给出理由，以便用于论文中。
```

### 17. 论文长文本处理

```
Prompt: Please read and polish the entire paper to ensure consistency and coherence.
提示：请阅读并润色整篇论文，确保一致性和连贯性。（就是这么简单粗暴！）

Prompt: I have written the XXX section, but I am not satisfied with its structure and coherence. Please help me reorganize the content and improve its coherence.
提示：我写了XXX部分，但我对其结构和连贯性不满意。请帮我重新组织内容，提高其连贯性。 

Prompt: Please review and revise the entire literature review section of my paperensuring that it meets the standards of academic writing and the content iscoherent and well-structured.
提示：请审查并修改我论文的整个文献综述部分，确保符合学术写作标准，内容连贯且结构合理。 
```

### 18. 提供独特见解

```
Prompt: Please provide me with some unique insights that I can discuss in my paper, based on the latest research that you are aware of.
提示：请根据你所了解的最新研究，为我提供一些独特的见解以便我在论文中进行讨论。 
```

### 19.评估

```
Prompt: Please help me to conduct an in-depth analysis of these research methods and data, and provide me with an assessment of their advantages and disadvantages.
提示：请帮助我对这些研究方法和数据进行深度分析，并为我提供关于其优缺点的评估。
```



## 3. ChatPaper

项目地址：https://github.com/kaixindelele/ChatPaper

### chatpaper.py

如果要用flask打开的话 ，要加东西https://github.com/kaixindelele/ChatPaper/pull/239/files

```python
from PIL import Image

PaperParams= namedtuple(
    "PaperParams",
    [
        "pdf_path",
        "query",
        "key_word",
        "filter_keys",
        "max_results",
        "sort",
        "save_image",
        "file_format",
        "language"
    ],
)

class Paper:
```

```python
用法: chat_paper.py [-h] [--pdf_path PATH] [--query QUERY] [--key_word KEYWORD]
                     [--language LANGUAGE] [--file_format FORMAT]
                     [--save_image SAVE_IMAGE] [--sort SORTCRITERIA]
                     [--max_results MAXRESULTS] [--filter_keys FILTERKEYS]
--pdf_path：指定本地 PDF 文档的路径，供脚本读取。如果未设置，脚本将直接从 arXiv 搜索并下载。
--query：ChatPaper 用于在 arXiv 上搜索论文的查询字符串。
查询字符串可以是以下格式：ti: xx, au: xx, all: xx,  ，其中 ti 表示标题，au 表示作者，all 表示所有字段。
例如，ti: chatgpt, au: robot 表示搜索标题包含 chatgpt 且作者包含 robot 的论文。
ti	标题
au	作者
abs	摘要
co	评论
jr	期刊引用
cat	主题类别
rn	报告编号
id	ID（请改用 id_list）
all	以上所有

--key_word：用户研究领域的关键词。该参数用于过滤与用户研究领域无关的论文。
例如，如果用户对强化学习感兴趣，他/她可以将 --key_word 设置为 reinforcement learning，这样 ChatPaper 将只总结与强化学习相关的论文。
--language：摘要的语言。目前，ChatPaper 支持两种语言：中文和英文。默认语言为中文。要使用英文，请将 --language 设置为 en。
--file_format：导出文件的格式。目前，ChatPaper 支持两种格式：Markdown 和纯文本。默认格式为 Markdown。要使用纯文本，请将 --file_format 设置为 txt。
--save_image：是否保存论文中的图片。保存一张图片需要一两分钟的时间。
--sort：搜索结果的排序标准。目前，ChatPaper 支持两种排序标准：相关性和最后更新日期。默认排序标准为相关性。要使用最后更新日期，请将 --sort 设置为 LastUpdatedDate。
--max_results：结果的最大数量。默认值为 1。
--filter_keys：过滤关键词。ChatPaper 仅会总结摘要中包含所有过滤关键词的论文。例如，如果用户对强化学习感兴趣，他/她可以将 --filter_keys 设置为 reinforcement learning，这样 ChatPaper 将只总结摘要中包含 reinforcement learning 的论文。
```
```python
[--pdf_path 是否直接读取本地的pdf文档？如果不设置的话，直接从arxiv上搜索并且下载] 
[--query 向arxiv网站搜索的关键词，有一些缩写示范：all, ti(title), au(author)，一个query示例：all: ChatGPT robot] 
[--key_word 你感兴趣领域的关键词，重要性不高] 
[--filter_keys 你需要在摘要文本中搜索的关键词，必须保证每个词都出现，才算是你的目标论文] 
[--max_results 每次搜索的最大文章数，经过上面的筛选，才是你的目标论文数，chat只总结筛选后的论文] 
[--sort arxiv的排序方式，默认是相关性，也可以是时间，arxiv.SortCriterion.LastUpdatedDate 或者 arxiv.SortCriterion.Relevance， 别加引号] 
[--save_image 是否存图片，如果你没注册gitee的图床的话，默认为false] 
[--file_format 文件保存格式，默认是markdown的md格式，也可以是txt] 

parser.add_argument("--pdf_path", type=str, default='', help="if none, the bot will download from arxiv with query")
parser.add_argument("--query", type=str, default='all: ChatGPT robot', help="the query string, ti: xx, au: xx, all: xx,")    
parser.add_argument("--key_word", type=str, default='reinforcement learning', help="the key word of user research fields")
parser.add_argument("--filter_keys", type=str, default='ChatGPT robot', help="the filter key words, 摘要中每个单词都得有，才会被筛选为目标论文")
parser.add_argument("--max_results", type=int, default=1, help="the maximum number of results")
parser.add_argument("--sort", default=arxiv.SortCriterion.Relevance, help="another is arxiv.SortCriterion.LastUpdatedDate")    
parser.add_argument("--save_image", default=False, help="save image? It takes a minute or two to save a picture! But pretty")
parser.add_argument("--file_format", type=str, default='md', help="导出的文件格式，如果存图片的话，最好是md，如果不是的话，txt的不会乱")
```

```python
# 使用 ChatPaper 在 arXiv 上进行批量搜索，并下载相关论文并生成摘要
python chat_paper.py --query "chatgpt robot" --filter_keys "chatgpt robot" --max_results 3
# 更加准确的
python chat_arxiv.py --query "chatgpt robot" --page_num 2 --max_results 3 --days 10

上述命令将在 arXiv 上搜索与 “chatgpt robot” 相关的论文，下载论文，并为每篇论文生成摘要。下载的 PDF 文件将保存在 ./pdf_files 文件夹中，摘要将保存在 ./export 文件夹中。
```

```python
# Arxiv在线批量搜索+下载+总结+高级搜索
python chat_paper.py --query "all: reinforcement learning robot 2023" --filter_keys "reinforcement robot" --max_results 3
# Arxiv在线批量搜索+下载+总结+高级搜索+指定作者
python chat_paper.py --query "au: Sergey Levine" --filter_keys "reinforcement robot" --max_results 3
```

```python
# 本地pdf总结
python chat_paper.py --pdf_path "demo.pdf"
# 本地文件夹批量总结
python chat_paper.py --pdf_path "your_absolute_path"
```

### chat_arxiv.py

```python
# 独有参数
parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, default='GPT-4', help="the query string, ti: xx, au: xx, all: xx,")
parser.add_argument("--key_word", type=str, default='GPT robot', help="the key word of user research fields")
parser.add_argument("--page_num", type=int, default=1, help="the maximum number of page")
parser.add_argument("--max_results", type=int, default=1, help="the maximum number of results")
parser.add_argument("--days", type=int, default=1, help="the last days of arxiv papers of this query")
parser.add_argument("--sort", type=str, default="web", help="another is LastUpdatedDate")
parser.add_argument("--save_image", default=False,
                    help="save image? It takes a minute or two to save a picture! But pretty")
parser.add_argument("--file_format", type=str, default='md', help="导出的文件格式，如果存图片的话，最好是md，如果不是的话，txt的不会乱")
parser.add_argument("--language", type=str, default='zh', help="The other output lauguage is English, is en")
```

```python
python chat_arxiv.py --query "chatgpt robot" --page_num 2 --max_results 3 --days 10

上述命令将在 arXiv 上搜索与 “chatgpt robot” 相关的论文，下载论文，并为每篇论文生成摘要。下载的 PDF 文件将保存在 ./pdf_files 文件夹中，摘要将保存在 ./export 文件夹中。
```

### chat_response.py

```python
parser = argparse.ArgumentParser()
parser.add_argument("--comment_path", type=str, default='review_comments.txt', help="path of comment")
parser.add_argument("--file_format", type=str, default='txt', help="output file format")
parser.add_argument("--language", type=str, default='en', help="output lauguage, en or zh")
```

### chat_reviewer.py

```python
parser = argparse.ArgumentParser()
parser.add_argument("--paper_path", type=str, default='', help="path of papers")
parser.add_argument("--file_format", type=str, default='txt', help="output file format")
parser.add_argument("--research_fields", type=str, default='computer science, artificial intelligence and reinforcement learning', help="the research fields of paper")
parser.add_argument("--language", type=str, default='en', help="output lauguage, en or zh")
```



### google_scholar_spider.py

谷歌学术论文整理

请参考 https://github.com/JessyTsu1/google_scholar_spider 了解具体用法和参数。

你还可以使用 `google_scholar_spider.py` 脚本在 Google Scholar 上进行批量搜索。例如，你可以使用以下命令在 Google Scholar 上搜索与 “deep learning” 相关的论文，并将结果保存到 `CSV` 文件中：

```
可以通过运行命令行中的google_scholar_spider函数并传递任何所需的参数来使用Google Scholar Spider。可用的参数包括：

--kw (default "machine learning") 要搜索的关键字。
--nresults (default 50) 要在Google Scholar上搜索的文章数。
--notsavecsv 使用此标志以不保存结果到CSV文件的方式打印结果。
--csvpath 要保存导出的CSV文件的路径。默认为当前文件夹。
--sortby (default "Citations") 按列排序数据。如果要按每年引用次数排序，请使用--sortby "cit/year"。
--plotresults 使用此标志以原始排名在x轴上，引用次数在y轴上绘制结果。
--startyear 搜索文章的起始年份。
--endyear (default current year) 搜索文章的结束年份。
--debug 使用此标志启用调试模式。调试模式用于单元测试并将页面存储在网络档案库中。
```

```python
python google_scholar_spider.py --kw "deep learning" --nresults 30 --csvpath "./data" --sortby "cit/year" --plotresults 1
# 这个命令在 Google Scholar 上搜索与 “deep learning” 相关的论文，检索 30 个结果，将结果保存到 ./data 文件夹中的 CSV 文件中，按每年的引用排序，并绘制结果。
```

### 网页版

```python
.\venv\Scripts\activate.bat
python app.py
```

arxiv  搜索 Arxiv 上的论文

```
参数：query, key_word, page_num, 
max_results, days, sort, save_image, 
file_format, language

示例：http://127.0.0.1:5000/arxiv?query=GPT-4&key_word=GPT+robot&page_num=1&max_results=1&days=1&sort=web&save_image=False&file_format=md&language=zh
```

paper  搜索并分析论文

```
参数：pdf_path, query, key_word, filter_keys, 
max_results, sort, save_image, 
file_format, language

示例：http://127.0.0.1:5000/paper?pdf_path=&query=all:+ChatGPT+robot&key_word=reinforcement+learning&filter_keys=ChatGPT+robot&max_results=1&sort=Relevance&save_image=False&file_format=md&language=zh
```

response  处理论文审稿评论。

```
参数：comment_path（要回复的审稿文本文件的路径）, file_format, language（使用英语，设置为 en。）

示例：http://127.0.0.1:5000/response?comment_path=review_comments.txt&file_format=txt&language=en
```

reviewer 查找论文审稿人。

```
参数：paper_path(要回复的审稿文本文件的路径), file_format, research_fields, language(要使用中文 zh)

示例：http://127.0.0.1:5000/reviewer?paper_path=&file_format=txt&research_fields=computer+science,+artificial+intelligence+and+reinforcement+learning&language=en
```

## 4. Gpt学术增强版

https://github.com/binary-husky/gpt_academic

为ChatGPT/GLM提供图形交互界面，特别优化论文阅读/润色/写作体验，模块化设计，支持自定义快捷按钮&函数插件，支持Python和C++等项目剖析&自译解功能，PDF/LaTex论文翻译&总结功能，支持并行问询多种LLM模型，支持清华chatglm等本地模型。兼容复旦MOSS, llama, rwkv, 盘古, newbing, claude等

```
conda activate gotac
python main.py
```



## 5. 精读论文AI

https://typeset.io/



## 6. ChatPPT

```
你作为一个拥有十年经验的PPT创作者，输出一份主题为《如何使用ai做ppt》的大纲，至少包含四个一级标题

请给每个章节搭配长相应的图片，包括所有的大章节和小章节，我希望保留上述所有大纲，返回图片用markdown格式，不要使用代码框和下划线等，使用unsplash API,地址是：https://source.unplash.com/1920×1080/?<关键词>

把上面的段落全部转换为markdown格式用代码输出
```

## 7. AI做视频

