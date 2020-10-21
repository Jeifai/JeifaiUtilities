package main

import (
    "os"
    "fmt"
    "encoding/csv"
    "github.com/gocolly/colly/v2"
)

var scraped_urls []string

func main() {
    base_page_url := "https://jobboersencheck.de/jobboersen-vergleich?page=%d"
    for i := 1; i < 34; i++ {
        extractJobBoersenCheckScraper(fmt.Sprintf(base_page_url, i))
    }
}

func extractJobBoersenCheckScraper(page_url string) {

    c := colly.NewCollector()
    c.OnHTML(".SearchResults-entry", func(e *colly.HTMLElement) {
        var data [][]string
        je_url := "https://jobboersencheck.de" + e.ChildAttr(".SearchResults-logo a", "href")
        je_name_1 := e.ChildAttr(".SearchResults-logo--image img", "title")
        je_name_2 := e.ChildText(".SearchResults-logo--text a")
        je_rating_count := e.ChildText(".SearchResults-rating-count")
        je_satisfaction := e.ChildText(".StarRating-rating")
        je_pricing := e.ChildText(".currency")
        row := []string{je_url, je_name_1 + je_name_2, je_rating_count, je_satisfaction, je_pricing, page_url}
        data = append(data, row)
        SaveRowToCsv("jobBoersenCheck/jobBoersenCheck.csv", data)
    })
    c.OnRequest(func(r *colly.Request) {
        fmt.Println(r.URL.String())
    })
    c.OnError(func(r *colly.Response, err error) {
        fmt.Println("ERROR")
    })
    c.Visit(page_url)
    return
}

func SaveRowToCsv(filename string, data [][]string) {
    f, err := os.OpenFile(filename, os.O_APPEND|os.O_WRONLY, os.ModeAppend)
    if err != nil {
        panic(err.Error())
    }
    defer f.Close()
    w := csv.NewWriter(f)
    w.Comma = ';'
    w.WriteAll(data)
    if err := w.Error(); err != nil {
        panic(err.Error())
    }
}
