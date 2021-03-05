from selenium import webdriver
import time
import re
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


def scroll_down(f_browser, url):
    f_browser.get(url)
    last_height = f_browser.execute_script("return document.body.scrollHeight")
    while True:
        f_browser.execute_script("window.scrollTo(0, document.body.scrollHeight-1000);")
        time.sleep(2)
        new_height = f_browser.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break
        last_height = new_height


comment_count = 0
positive_comment_count = 0
negative_comment_count = 0

PAGE_COMMENT_COUNT = 30
PAGE_DEFAULT_STRING = "?section=comments&page="

browser = webdriver.Firefox()

regions = ["https://www.yemeksepeti.com/izmir/dokuz-eylul-universitesi-tinaztepe-kampusu#ors:false"]

with open('Comments.txt', 'w', encoding="utf-8") as file:
    for region in regions:
        print(region + " started.")
        browser.get(region)
        scroll_down(browser, region)
        time.sleep(3)

        elements = browser.find_elements_by_css_selector(".restaurantName")

        urls = list()

        for i in elements:
            current_url = i.get_attribute('href')
            urls.append(current_url)

        for i in range(len(urls)):
            try:
                print("Restaurant " + str(i + 1) + "/" + str(len(urls)))
                current_url = urls[i]
                browser.get(urls[i])
                if "closed" in browser.current_url:
                    element = WebDriverWait(browser, 20).until(
                        ec.element_to_be_clickable((By.XPATH, "/html/body/div[10]/div["
                                                              "2]/div/div/div[1]/div["
                                                              "2]/img")))
                    element.click()

                element = browser.find_element(By.XPATH, "/html/body/div[3]/div/div[2]/div/div[2]/div[1]/ul/li[4]/a")
                total_comments_size = re.split('\\s+', element.text)
                total_comments_size = int(total_comments_size[1][1:-1])

                if total_comments_size % PAGE_COMMENT_COUNT == 0:
                    page_count = int(total_comments_size / PAGE_COMMENT_COUNT)
                else:
                    page_count = round((total_comments_size / PAGE_COMMENT_COUNT) + 0.5)
                element.click()

                comments = browser.find_elements_by_css_selector(".comment.row")
                points = browser.find_elements_by_css_selector(".restaurantPoints.col-md-12")

                iterator = 0
                point_iterator = 0
                print("Page 1/" + str(page_count))
                while iterator < len(comments):

                    comment = comments[iterator].text
                    comment = comment.splitlines()
                    if (len(comment) == 2) and (comment[1] != '['):
                        temp = re.split(': | \\| ', points[point_iterator].text)
                        sentence = comment[1] + "#" + temp[1] + "#" + temp[3] + "#" + temp[5] + "\n"
                        ara_puan = (int(temp[1]) + int(temp[3]) + int(temp[5])) / 3  # yeni satır
                        if ara_puan < 7:  # yeni satır
                            file.write(sentence)
                            comment_count += 1
                        point_iterator += 1
                    iterator += 1

                if page_count != 1:
                    for m in range(2, page_count + 1):

                        browser.get(current_url + PAGE_DEFAULT_STRING + str(m))
                        if "closed" in browser.current_url:
                            element = WebDriverWait(browser, 20).until(
                                ec.element_to_be_clickable((By.XPATH, "/html/body/div[10]/div["
                                                                      "2]/div/div/div[1]/div["
                                                                      "2]/img")))
                            element.click()

                        print("Page " + str(m) + "/" + str(page_count))

                        comments = browser.find_elements_by_css_selector(".comment.row")
                        points = browser.find_elements_by_css_selector(".restaurantPoints.col-md-12")
                        iterator = 0
                        point_iterator = 0
                        while iterator < len(comments):
                            comment = comments[iterator].text
                            comment = comment.splitlines()
                            if (len(comment) == 2) and (comment[1] != '['):
                                temp = re.split(': | \\| ', points[point_iterator].text)
                                sentence = comment[1] + "#" + temp[1] + "#" + temp[3] + "#" + temp[5] + "\n"
                                ara_puan = (int(temp[1]) + int(temp[3]) + int(temp[5])) / 3  # yeni satır
                                if ara_puan < 7:  # yeni satır
                                    file.write(sentence)
                                    comment_count += 1
                                point_iterator += 1
                            iterator += 1
            except Exception:
                print("Exception")
                continue

    print("Ended.\nComment Count: " + str(comment_count))

    browser.close()
