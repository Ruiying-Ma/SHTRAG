import os
import re
from typing import Dict, List, Set
import fitz
import logging
from .utils import clean_space, max_min
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.DEBUG)

FEATURE_EXTRACTOR_DEBUG = True

class FeatureExtractor:
    '''
    Take a PDF and its headings & list items as inputs. Extract their visual patterns. 
    '''
    def __init__(self, thresh_is_centered, round_font_size, thresh_is_underlined, thresh_is_line, thresh_rect):
        self.pdf_doc = None
        self.page = None
        self.rect = None
        self.underlines = None
        self.thresh_is_centered = thresh_is_centered
        self.round_font_size = round_font_size
        self.thresh_is_underlined = thresh_is_underlined
        self.thresh_is_line = thresh_is_line
        self.thresh_rect = thresh_rect
        # refer to https://github.com/stanfordnlp/pdf-struct/blob/main/pdf_struct/features/listing/en.py
        self._LIST_TEMPLATES = [
            '^[ \t]*\([ \t]*{num}[ \t]*\)', 
            '^[ \t]*{num}[ \t]*\. ', 
            '^[ \t]*{num}[ \t]*\)', 
            '^[ \t]*§[ \t]*{num}', 
            '^[ \t]*{num}[ \t]*:', 
            '^[ \t]*ARTICLE[ \t]*{num}', 
            '^[ \t]*Article[ \t]*{num}', 
            '^[ \t]*SECTION[ \t]*{num}', 
            '^[ \t]*Section[ \t]*{num}', 
            '^[ \t]*Item[ \t]*{num}', 
            '^[ \t]*ITEM[ \t]*{num}',
            ]
        self._m_list_type_to_regex = {
            "alpha_lower": "(?P<num>[a-z])",
            "alpha_upper": "(?P<num>[A-Z])",
            "numeric": "(?P<num>[1-9][0-9]*)",
            "roman_upper": "(?P<num>(?=.)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))", # https://dev.to/alexdjulin/a-python-regex-to-validate-roman-numerals-2g99
            "roman_lower": "(?P<num>(?=.)m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))",
            "numeric_multilevel": "(?P<num>[1-9][0-9]*(?:[\.-][1-9][0-9]*)*[\.-][1-9][0-9]*)",
            "bullet": '(?P<num>[・\-*+•‣⁃○∙◦⦾⦿\uE000-\uF8FF\u0083\u00be\u2022])',
        }

        self._m_list_type_to_patterns = dict()
        for list_type, regex in self._m_list_type_to_regex.items():
            if list_type == "bullet":
                tmpl = '^[ \t]*{num}'
                self._m_list_type_to_patterns[list_type] = [re.compile(tmpl.format(num=regex))]
            else:
                assert list_type in ["numeric", "roman_upper", "roman_lower", "alpha_upper", "alpha_lower", "numeric_multilevel"]
                self._m_list_type_to_patterns[list_type] = [re.compile(tmpl.format(num=regex)) for tmpl in self._LIST_TEMPLATES]

    def _alpha_to_int(self, alpha_expr: str) -> int:
        '''
        Convert an alphabetic numbering to 1-base integer. Refer to https://github.com/stanfordnlp/pdf-struct/blob/main/pdf_struct/features/listing/en.py

        Args:
            - alpha_expr (str)

        Returns:
            - integer numbering
        '''
        if not isinstance(alpha_expr, str):
            raise ValueError("alpha_expr must be string")

        pattern = re.compile(r'[a-z]')
        if not bool(pattern.fullmatch(alpha_expr.lower())):
            raise ValueError("alpha_expr has wrong form")

        return ord(alpha_expr.lower()) - ord("a") + 1

    def _roman_to_int(self, roman_expr: str) -> int:
        '''
        Convert a roman numbering to 1-base integer. Refer to https://github.com/stanfordnlp/pdf-struct/blob/main/pdf_struct/features/listing/en.py

        Args:
            - roman_expr (str)

        Return:
            - integer numbering
        '''
        if not isinstance(roman_expr, str):
            raise ValueError("roman_expr must be string")
            
        pattern = re.compile(r'm{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})')
        if not bool(pattern.fullmatch(roman_expr.lower())):
            raise ValueError("roman_expr has wrong form")

        expr = roman_expr.upper()
        m_char_to_int = {'M':1000, 'D':500, 'C':100, 'L':50, 'X':10, 'V':5, 'I':1}
        sum = 0
        for i in range(len(expr)):
            try:
                value = m_char_to_int[expr[i]]
                # If the next place holds a larger number, this value is negative
                if i + 1 < len(expr) and m_char_to_int[expr[i + 1]] > value:
                    sum -= value
                else:
                    sum += value
            except KeyError:
                raise ValueError(f"Invalid roman expression: {roman_expr}")
        return sum

    def load_pdf_doc(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise ValueError("pdf_path doesn't exist")

        if self.pdf_doc is not None:
            raise ValueError("pdf_doc already loaded")

        logging.info(f"Loading pdf {os.path.basename(pdf_path)} to FeatureExtractor...")

        self.pdf_doc: fitz.Document = fitz.open(pdf_path)

    def _clear_page_and_underlines(self):
        '''
        Set page and underlines as None
        '''
        self.page = None
        self.underlines = None

    def load_page_and_underlines(self, page_number: int):
        '''
        Load a page and extract all underlines from the self.pdf_doc. Note that in fitz, page number is 0-based, while in pdf-document-layout-analysis, page number if 1-based.

        Args:
            - page_number: 1-based page number returned by pdf-document-layout-analysis
        '''
        if self.pdf_doc is None:
            raise ValueError("Please load pdf_doc before extracting font info")
        elif not isinstance(self.pdf_doc, fitz.Document):
            raise ValueError("pdf_doc must be an instance of fitz.Document")

        if not isinstance(page_number, int) or page_number < 1 or page_number > self.pdf_doc.page_count:
            raise ValueError("page_number must be an integer in range [1, page_count]")
            
        if self.page is not None and self.page.number == page_number - 1:
            return
        
        self._clear_page_and_underlines()

        logging.info(f"Loading page {page_number} and its underlines to FeatureExtractor...")
        self.page: fitz.Page = self.pdf_doc.load_page(page_id=page_number - 1)

        self.underlines: List[fitz.Rect] = []
        path_dicts_list = self.page.get_drawings()
        for path_dict in path_dicts_list:
            for item in path_dict["items"]:
                # item is a list of tuples, where item[0] is the drawing command
                draw_command = item[0] # "l" = line, "re" = rect, "qu" = quad, "c" = curve. See https://pymupdf.readthedocs.io/en/latest/recipes-drawing-and-graphics.html#how-to-extract-drawings
                if draw_command == "l":
                    # item[1] = left_point
                    # item[2] = right_point
                    left_point: fitz.Point = item[1]
                    right_point: fitz.Point = item[2]
                    if abs(left_point.y - right_point.y) <= self.thresh_is_line:
                        self.underlines.append(
                            fitz.Rect(
                                x0=min(left_point.x, right_point.x),
                                y0=min(left_point.y, right_point.y),
                                x1=max(left_point.x, right_point.x),
                                y1=max(left_point.y, right_point.y),
                            )
                        )
                elif draw_command == "re":
                    # item[1] = rect
                    rect: fitz.Rect = item[1]
                    if rect.height <= self.thresh_is_line:
                        self.underlines.append(item[1])

    def _clear_rect(self):
        '''
        Set rect as None
        '''
        self.rect = None

    def load_rect(self, left, top, width, height, page_number):
        '''
        Load a fitz.Rect from page. Note that in fitz.Rect, 
            - x0 = left
            - y0 = top
            - x1 = right
            - y1 = bottom
        and x0 < x1, y0 < y1. 
        
        Args:
            - page_number: 1-based page number returned by pdf-document-layout-analysis
            - left, top, width, height: floats returned by pdf-document-layout-analysis
        '''
        if self.pdf_doc is None:
            raise ValueError("Please load pdf_doc before extracting font info")
        elif not isinstance(self.pdf_doc, fitz.Document):
            raise ValueError("pdf_doc must be an instance of fitz.Document")
        
        if self.page is None:
            raise ValueError("Please load page before extracting font info")
        elif not isinstance(self.page, fitz.Page):
            raise ValueError("page must be an instance of fitz.Page")

        if self.underlines is None:
            raise ValueError("Please load underlines before extracting font info")
        elif (not isinstance(self.underlines, List)) or (not all([isinstance(u, fitz.Rect) for u in self.underlines])):
            raise ValueError("underlines must be an instance of List[fitz.Rect]")
            
        if not isinstance(page_number, int) or page_number < 1 or page_number > self.pdf_doc.page_count:
            raise ValueError("page_number must be an integer in range [1, page_count]")
        elif page_number - 1 != self.page.number:
            raise ValueError("Current page in FeatureExtractor is not the wanted page")

        if not isinstance(left, float) or not isinstance(top, float) or not isinstance(width, float) or not isinstance(height, float):
            raise ValueError("left, top, width, height must be float numbers")

        self._clear_rect()

        logging.info(f"Loading rect({left}, {top}, {width}, {height}) (l, t, w, h) to FeatureExtractor...")
        self.rect: fitz.Rect = fitz.Rect(
            x0=max(left - self.thresh_rect, 0),
            y0=max(top - self.thresh_rect, 0),
            x1=left + width + self.thresh_rect,
            y1=top + height + self.thresh_rect,
        )

    def extract_font_info(self) -> Dict:
        '''
        Extract font size, name, color from self.rect. For rect info, see https://pymupdf.readthedocs.io/en/latest/textpage.html#textpagedict

        Returns:
            - font_info (Dict)
                - font_size (float, 2)
                - font_name (str)
                - font_color (int)
        '''

        if self.pdf_doc is None:
            raise ValueError("Please load pdf_doc before extracting font info")
        elif not isinstance(self.pdf_doc, fitz.Document):
            raise ValueError("pdf_doc must be an instance of fitz.Document")
        
        if self.page is None:
            raise ValueError("Please load page before extracting font info")
        elif not isinstance(self.page, fitz.Page):
            raise ValueError("page must be an instance of fitz.Page")

        if self.underlines is None:
            raise ValueError("Please load underlines before extracting font info")
        elif (not isinstance(self.underlines, List)) or (not all([isinstance(u, fitz.Rect) for u in self.underlines])):
            raise ValueError("underlines must be an instance of List[fitz.Rect]")

        if self.rect is None:
            raise ValueError("Please load rect before extracting font info")
        elif not isinstance(self.rect, fitz.Rect):
            raise ValueError("rect must be an instance of fitz.Rect")

        page_dict = self.page.get_text("dict", clip=self.rect)
        # print(self.page.get_text(clip=self.rect))###############################
        # print(len(self.underlines))############################

        block_dict_list = page_dict["blocks"]

        m_features_to_stats = {
            "font_size": dict(),
            "font_name": dict(),
            "font_color": dict(),
        }

        for block_dict in block_dict_list:
            if block_dict["type"] == 1: # image block
                continue
            else:
                assert block_dict["type"] == 0 # text block
                line_dict_list = block_dict["lines"]
                for line_dict in line_dict_list:
                    span_dict_list= line_dict["spans"]
                    for span_dict in span_dict_list:
                        feature_info = {
                            "font_size": round(span_dict["size"], self.round_font_size),
                            "font_name": span_dict["font"],
                            "font_color": span_dict["color"],
                        }
                        for feature_key, feature_val in feature_info.items():
                            if feature_val not in m_features_to_stats[feature_key]:
                                m_features_to_stats[feature_key][feature_val] = max(1, len(span_dict["text"]))
                            else:
                                m_features_to_stats[feature_key][feature_val] += max(1, len(span_dict["text"]))
        
        font_info = dict()
        for feature in m_features_to_stats:
            stats = m_features_to_stats[feature]
            stats_list = [(k, v) for k, v in stats.items()]
            font_info[feature] = min(stats_list, key=lambda kv: (-kv[1], kv[0]))[0]
            # check
            if FEATURE_EXTRACTOR_DEBUG:
                for k in stats:
                    assert stats[k] <= stats[font_info[feature]]

        # check
        if FEATURE_EXTRACTOR_DEBUG:
            for feature in m_features_to_stats:
                assert feature in font_info
            assert isinstance(font_info["font_size"], float)
            assert isinstance(font_info["font_name"], str)
            assert isinstance(font_info["font_color"], int)

        return font_info

    def extract_clean_text(self) -> str:
        '''
        Extract text from rect and clean the spaces.
        '''
        if self.pdf_doc is None:
            raise ValueError("Please load pdf_doc before extracting font info")
        elif not isinstance(self.pdf_doc, fitz.Document):
            raise ValueError("pdf_doc must be an instance of fitz.Document")
        
        if self.page is None:
            raise ValueError("Please load page before extracting font info")
        elif not isinstance(self.page, fitz.Page):
            raise ValueError("page must be an instance of fitz.Page")

        if self.underlines is None:
            raise ValueError("Please load underlines before extracting font info")
        elif (not isinstance(self.underlines, List)) or (not all([isinstance(u, fitz.Rect) for u in self.underlines])):
            raise ValueError("underlines must be an instance of List[fitz.Rect]")

        if self.rect is None:
            raise ValueError("Please load rect before extracting font info")
        elif not isinstance(self.rect, fitz.Rect):
            raise ValueError("rect must be an instance of fitz.Rect")

        return clean_space(self.page.get_text("text", clip=self.rect))

    def is_centered(self) -> bool:
        if self.pdf_doc is None:
            raise ValueError("Please load pdf_doc before extracting font info")
        elif not isinstance(self.pdf_doc, fitz.Document):
            raise ValueError("pdf_doc must be an instance of fitz.Document")
        
        if self.page is None:
            raise ValueError("Please load page before extracting font info")
        elif not isinstance(self.page, fitz.Page):
            raise ValueError("page must be an instance of fitz.Page")

        if self.underlines is None:
            raise ValueError("Please load underlines before extracting font info")
        elif (not isinstance(self.underlines, List)) or (not all([isinstance(u, fitz.Rect) for u in self.underlines])):
            raise ValueError("underlines must be an instance of List[fitz.Rect]")

        if self.rect is None:
            raise ValueError("Please load rect before extracting font info")
        elif not isinstance(self.rect, fitz.Rect):
            raise ValueError("rect must be an instance of fitz.Rect")
            
        page_width = self.page.rect.width
        page_mid = page_width / 2
        rect_mid = self.rect / 2
        return abs(rect_mid - page_mid) <= self.thresh_is_centered

    def is_all_cap(self, text) -> bool:
        if not isinstance(text, str):
            raise ValueError("text must be a string")

        for char in text:
            if char.isalpha() and (not char.isupper()):
                return False
        return True

    def extract_list_type(self, texts_list: List[str]) -> List:
        '''
        Extract list type of a list of texts. Refer to https://github.com/stanfordnlp/pdf-struct/blob/main/pdf_struct/features/listing/en.py

        Args:
            - texts_list (List[str])

        Returns:
            - a list of "numeric", "roman_upper", "roman_lower", "alpha_upper", "alpha_lower", "numeric_multilevel_{level_num}", "bullet_{bullet_type}", "none"
        '''
        if (not isinstance(texts_list, List)) or (not all([isinstance(t, str) for t in texts_list])):
            raise ValueError("texts_list must be a list of strings")

        m_alpha_roman_to_numberings_list = {
            "alpha_upper": [],
            "alpha_lower": [],
            "roman_upper": [],
            "roman_lower": [],
        }

        list_types_list = []

        for text in texts_list:
            m_list_type_to_numbering = dict()
            for list_type, patterns_list in self._m_list_type_to_patterns.items():
                for pattern in patterns_list:
                    m = pattern.match(text)
                    if m is not None:
                        assert list_type not in m_list_type_to_numbering
                        if m.group("num").strip() != "":
                            m_list_type_to_numbering[list_type] = m.group("num")

            list_type = None
            if len(m_list_type_to_numbering) == 0:
                list_type = "none"
            elif len(m_list_type_to_numbering) == 1:
                list_type = list(m_list_type_to_numbering.keys())[0]
                if "alpha" in list_type:
                    assert list_type in m_alpha_roman_to_numberings_list
                    int_numbering = self._alpha_to_int(m_list_type_to_numbering[list_type])
                elif "roman" in list_type:
                    assert list_type in m_alpha_roman_to_numberings_list
                    int_numbering = self._roman_to_int(m_list_type_to_numbering[list_type])
                if list_type in m_alpha_roman_to_numberings_list:
                    m_alpha_roman_to_numberings_list[list_type].append(int_numbering)
            elif len(m_list_type_to_numbering) == 2:
                if "numeric" in m_list_type_to_numbering and "numeric_multilevel" in m_list_type_to_numbering:
                    list_type = "numeric_multilevel"
                elif ("alpha_upper" in m_list_type_to_numbering and "roman_upper" in m_list_type_to_numbering) or ("alpha_lower" in m_list_type_to_numbering and "roman_lower" in m_list_type_to_numbering):
                    alpha_list_type = [key for key in m_list_type_to_numbering if "alpha" in key][0]
                    roman_list_type = [key for key in m_list_type_to_numbering if "roman" in key][0]
                    alpha_numbering = self._alpha_to_int(m_list_type_to_numbering[alpha_list_type])
                    roman_numbering = self._roman_to_int(m_list_type_to_numbering[roman_list_type])

                    pre_alpha_numbering = max_min(m_alpha_roman_to_numberings_list[alpha_list_type], alpha_numbering)
                    pre_roman_numbering = max_min(m_alpha_roman_to_numberings_list[roman_list_type], roman_numbering)

                    assert alpha_numbering > pre_alpha_numbering and pre_alpha_numbering >= 0
                    assert roman_numbering > pre_roman_numbering and pre_roman_numbering >= 0

                    if alpha_numbering - pre_alpha_numbering > roman_numbering - pre_roman_numbering:
                        list_type = roman_list_type
                    elif alpha_numbering - pre_alpha_numbering < roman_numbering - pre_roman_numbering:
                        list_type = alpha_list_type
                    else:
                        if pre_alpha_numbering > 0 and pre_roman_numbering == 0:
                            list_type = alpha_list_type
                        elif pre_alpha_numbering == 0 and pre_roman_numbering > 0:
                            list_type = roman_list_type
                        else:
                            if any([(character in m_list_type_to_numbering[roman_list_type].lower()) for character in ["l", "c", "d", "m"]]):
                                list_type = alpha_list_type
                            else:
                                list_type = roman_list_type
                    if "alpha" in list_type:
                        m_alpha_roman_to_numberings_list[list_type].append(alpha_numbering)
                    else:
                        assert "roman" in list_type
                        m_alpha_roman_to_numberings_list[list_type].append(roman_numbering)
                    
                else:
                    raise ValueError(f"Invalid case for two candidate list types for '{text}': {m_list_type_to_numbering}")
            else:
                raise ValueError(f"Cannot exist more than 2 possible list types for '{text}': {m_list_type_to_numbering}")
            assert list_type is not None
            if list_type == "numeric_multilevel":
                level = len(re.split(r'[.-]', m_list_type_to_numbering[list_type]))
                list_type += f"_{level}"
            elif list_type == "bullet":
                list_type += f"_{m_list_type_to_numbering[list_type]}"
            list_types_list.append(list_type)

        assert len(list_types_list) == len(texts_list)
        return list_types_list

    def is_underlined(self) -> bool:
        if self.pdf_doc is None:
            raise ValueError("Please load pdf_doc before extracting font info")
        elif not isinstance(self.pdf_doc, fitz.Document):
            raise ValueError("pdf_doc must be an instance of fitz.Document")
        
        if self.page is None:
            raise ValueError("Please load page before extracting font info")
        elif not isinstance(self.page, fitz.Page):
            raise ValueError("page must be an instance of fitz.Page")

        if self.underlines is None:
            raise ValueError("Please load underlines before extracting font info")
        elif (not isinstance(self.underlines, List)) or (not all([isinstance(u, fitz.Rect) for u in self.underlines])):
            raise ValueError("underlines must be an instance of List[fitz.Rect]")

        if self.rect is None:
            raise ValueError("Please load rect before extracting font info")
        elif not isinstance(self.rect, fitz.Rect):
            raise ValueError("rect must be an instance of fitz.Rect")

        candid_underlines_num = 0

        for line_rect in self.underlines:
            left_margin = (abs(line_rect.x0 - self.rect.x0) <= self.thresh_is_underlined)
            right_margin = (abs(line_rect.x1 - self.rect.x1) <= self.thresh_is_underlined)
            horizontal_margin = (abs(line_rect.y0 - self.rect.y1) <= self.thresh_is_underlined)
            if left_margin and right_margin and horizontal_margin:
                candid_underlines_num += 1

        return (candid_underlines_num == 1)

    def reset(self):
        '''
        Reset FeatureExtractor:
        - self.pdf_doc = None
        - self.page = None
        - self.rect = None
        - self.underlines = None
        '''
        self.pdf_doc = None
        self.page = None
        self.rect = None
        self.underlines = None