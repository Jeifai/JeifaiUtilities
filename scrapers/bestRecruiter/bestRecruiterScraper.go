package main

import (
    "os"
    "fmt"
    "strings"
    "net/url"
    "encoding/xml"
    "encoding/csv"
    "github.com/gocolly/colly/v2"
)

func main() {
    start_url := "https://www.bestcruiter.com/sitemap.xml"
    xmlBestRecruiterScraper(start_url)
}

func xmlBestRecruiterScraper(start_url string) {
    type BestRecruiterXml struct {
        XMLName xml.Name `xml:"urlset"`
        Text    string   `xml:",chardata"`
        Xmlns   string   `xml:"xmlns,attr"`
        URL     []struct {
            Text       string `xml:",chardata"`
            Loc        string `xml:"loc"`
            Priority   string `xml:"priority"`
            Changefreq string `xml:"changefreq"`
            Lastmod    string `xml:"lastmod"`
        } `xml:"url"`
    }
    c := colly.NewCollector()
    c.OnResponse(func(r *colly.Response) {
        var bestRecruiterXml BestRecruiterXml
        err := xml.Unmarshal(r.Body, &bestRecruiterXml)
        if err != nil {
            panic(err.Error())
        }
        for _, elem := range bestRecruiterXml.URL {
            if strings.Contains(elem.Loc, "/person/") {
                if len(elem.Loc) > len("https://bestcruiter.com/person/") {
                    recruiter_url := elem.Loc
                    fmt.Println(recruiter_url)
                    extractBestRecruiterScraper(recruiter_url)
                }
            }
        }
    })
    c.Visit(start_url)
}

func extractBestRecruiterScraper(recruiter_url string) {
    c := colly.NewCollector()
    l := c.Clone()
    c.OnHTML(".profil-wrap", func(e *colly.HTMLElement) {
        var data [][]string
        recruiter_name := e.ChildText(".profil-name")
        company_name := e.ChildText(".profil-company")
        company_url := e.ChildAttr(".profil-company a", "href")
        location := e.ChildText(".profil-city")
        total_feedback := e.ChildText(".rates-mark-wrap")
        count_feedback := e.ChildText(".profil-rates p strong")
        row := []string{recruiter_name, company_name, company_url, location, total_feedback, count_feedback, recruiter_url}
        data = append(data, row)
        SaveRowToCsv("bestRecruiter/recruitersData.csv", data)
        _, err := url.ParseRequestURI(company_url)
        if err == nil {
           l.Visit(company_url)
        }

    })
    l.OnHTML(".profil-card.personalberatung", func(e *colly.HTMLElement) {
        var data [][]string
        company_name := e.ChildText(".profil-name")
        company_address := strings.Join(e.ChildTexts(".col-sm-5 div"), "|")
        company_socials := strings.Join(e.ChildAttrs(".social-box a", "href"), "|")
        company_contacts := strings.Join(e.ChildTexts(".col-xs-9"), "|")
        company_url := e.Request.URL.RawPath
        row := []string{company_name, company_address, company_socials, company_contacts, company_url}
        data = append(data, row)
        SaveRowToCsv("bestRecruiter/companiesData.csv", data)
        _, err := url.ParseRequestURI(company_url)
        if err == nil {
           l.Visit(company_url)
        }

    })
    c.OnRequest(func(r *colly.Request) {
        fmt.Println(r.URL.String())
    })
    c.OnError(func(r *colly.Response, err error) {
        fmt.Println(r.Request.URL)
    })
    l.OnRequest(func(r *colly.Request) {
        fmt.Println(r.URL.String())
    })
    l.OnError(func(r *colly.Response, err error) {
        fmt.Println(r.Request.URL)
    })
    c.Visit(recruiter_url)
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
