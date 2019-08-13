#!/usr/bin/env python3.7

from htmldoom import base as b
from htmldoom import elements as e
from htmldoom import render as _render
from htmldoom import renders

doctype = _render(
    b.doctype(
        'html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd"'
    )
)

contents = _render(
    e.div(id_="header")(e.h2()("Super Mario 64 Decompilation - Documentation Graph")),
    e.div(id_="content")(
        e.div(class_="demo-container")(
            e.div(id_="placeholder", class_="demo-placeholder")
        ),
        e.p()("Hello!"),
    ),
    e.div(id_="footer")(" Copyright © 2007 - 2014 IOLA and Ole Laursen "),
)


@renders(e.title()("Super Mario 64 Documentation Graph"))
def render_title(data):
    return {}


@renders(
    e.head()(
        e.meta(http_equiv="Content-Type", content="text/html; charset=utf-8"),
        "{title}",
        e.link(href="graph.css", rel="stylesheet", type_="text/css"),
        e.script(
            language="javascript", type_="text/javascript", src="js/jquery.min.js"
        ),
        *[
            e.script(
                language="javascript", type_="text/javascript", src=js_src + ".min.js"
            )
            for js_src in [
                "js/jquery.canvaswrapper",
                "js/jquery.colorhelpers",
                "js/jquery.flot",
                *[
                    "js/jquery.flot." + submodule
                    for submodule in [
                        "saturated",
                        "browser",
                        "drawSeries",
                        "errorbars",
                        "uiConstants",
                        "logaxis",
                        "symbol",
                        "flatdata",
                        "navigate",
                        "fillbetween",
                        "stack",
                        "touchNavigate",
                        "hover",
                        "touch",
                        "time",
                        "axislabels",
                        "selection",
                    ]
                ],
            ]
        ],
        e.script(type_="text/javascript")(
            """
            $(function() {{

                var total = {results};

		        $.plot(
                    "#placeholder",
                    [
                        {{
                            data: total,
                            lines: {{ show: true, steps: true }}
                        }}
                    ],
                    {{
                        xaxis: {{ mode: "time", timeBase: "milliseconds" }}
		            }}
                );

                // Add the Flot version string to the footer
                $("#footer").prepend("Flot " + $.plot.version + " &ndash; ");
            }});
            """.encode(
                "utf-8"
            )
        ),
    )
)
def render_head(data, title_renderer=render_title):
    return {"title": title_renderer(data=data), "results": data["results"]}


@renders(e.body()("{contents}"))
def render_body(data):
    return {"contents": contents}


@renders(e.html()("{head}", "{body}"))
def render_html(
    data,
    title_renderer=render_title,
    head_renderer=render_head,
    body_renderer=render_body,
):
    return {
        "head": head_renderer(data=data, title_renderer=render_title),
        "body": body_renderer(data=data),
    }


@renders("{doctype}{html}")
def render_document(
    data,
    title_renderer=render_title,
    head_renderer=render_head,
    body_renderer=render_body,
    html_renderer=render_html,
):
    return {
        "doctype": doctype,
        "html": html_renderer(
            data=data,
            title_renderer=title_renderer,
            head_renderer=head_renderer,
            body_renderer=body_renderer,
        ),
    }


def render(data):
    return render_document(data=data)


if __name__ == "__main__":
    print(render({}))
