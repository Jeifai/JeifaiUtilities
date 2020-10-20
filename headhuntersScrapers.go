    package main

    import (
        "os"
        "fmt"
        "log"
        "strconv"
        "strings"
        "net/http"
        "io/ioutil"
        "encoding/csv"
        "github.com/gocolly/colly/v2"
    )

    func main() {
        for i := 0; i < 380; i++ {
            scrape(i)
        }
    }

    func scrape(page int) {
        results_per_page := 25
        file_name := fmt.Sprintf("headhunters_%d.html", page)
        url := fmt.Sprintf("https://eu.experteer.com/headhunter/refine_search?offset=%d&search[page_size]=25", results_per_page * page)
        base_url := "https://eu.experteer.com%s"
        client := &http.Client{}
        req, err := http.NewRequest("POST", url, nil)
        if err != nil {
            log.Fatal(err)
        }
        resp, err := client.Do(req)
        if err != nil {
            log.Fatal(err)
        }
        bodyText, err := ioutil.ReadAll(resp.Body)
        if err != nil {
            log.Fatal(err)
        }
        SaveResponseToFileWithFileName(string(bodyText), file_name)

        counter := 0
        c := colly.NewCollector()
        c.OnHTML("tbody", func(e *colly.HTMLElement) {
            var data [][]string
            var name, name_url, title, jobs, contacts, company, company_url, location, keywords string
            e.ForEach("td", func(_ int, el *colly.HTMLElement) {
                switch counter {
                    case 1:
                        name = el.ChildText(".search-result-profile-name a")
                        name_url = fmt.Sprintf(base_url, el.ChildAttr(".search-result-profile-name a", "href"))
                    case 2:
                        title = el.ChildText("p")
                    case 3:
                        jobs = el.ChildText("p")
                    case 4:
                        contacts = el.ChildText("p")
                    case 5:
                        company = el.ChildText("a")
                        company_url = fmt.Sprintf(base_url, el.ChildAttrs("a", "href")[0])
                    case 6:
                        location = el.ChildText("p")
                    case 7:
                        keywords = strings.Join(
                            strings.Fields(
                                strings.TrimSpace(
                                    strings.ReplaceAll(
                                        el.ChildText(".grid-data.is-2of3"), "\n", " "))), " ")
                }
                counter++
            })
            row := []string{strconv.Itoa(page), name, name_url, title, jobs, contacts, company, company_url, location, keywords}
            if name != "" {
                data = append(data, row)
                SaveRowToCsv(data)
            }
            counter = 0
        })
        c.OnRequest(func(r *colly.Request) {
            fmt.Println("Visiting ", r.URL.String())
        })   
        c.OnScraped(func(r *colly.Response) {
            RemoveFileWithFileName(file_name)
        })
        t := &http.Transport{}
        t.RegisterProtocol("file", http.NewFileTransport(http.Dir("/")))
        dir, err := os.Getwd()
        if err != nil {
            panic(err.Error())
        }
        c.WithTransport(t)
        c.Visit("file:" + dir + "/" + file_name) 
    }

    func SaveResponseToFileWithFileName(response string, filename string) {
        dir, err := os.Getwd()
        if err != nil {
            panic(err.Error())
        }
        f, err := os.Create(dir + "/" + filename)
        if err != nil {
            panic(err.Error())
        }
        defer f.Close()
        f.WriteString(response)
    }

    func RemoveFileWithFileName(filename string) {
        dir, err := os.Getwd()
        if err != nil {
            panic(err.Error())
        }
        err = os.Remove(dir + "/" + filename)
        if err != nil {
            panic(err.Error())
        }
    }

    func SaveRowToCsv(data [][]string) {
        var path = "headhunterdata.csv"
        f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, os.ModeAppend)
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
